"""soc_artifacts — contrat de données C9-4 et journal des décisions analyste (C9-8).

Module partagé, à l'identique, par le notebook (producteurs) et par l'application
(dashboard Streamlit, API) : un seul code écrit, vérifie et relit les artefacts.

Contrat (TODO NV132 rév. 11, validé le 07/11/2026) :
  Niveau A — métadonnées d'artefact (tout fichier : modèle, index, état, résultats).
  Niveau B — enregistrements par finding (finding_id obligatoire), seulement pour
             les résultats associés à un finding.
  payload_sha256 = SHA-256 des octets du SEUL fichier de données ; il est écrit
  dans un manifeste séparé (<kind>.manifest.json) qui n'entre pas dans le hash.
  À la lecture : schema_version connu, hash recalculé et identique, sinon refus.

Journal analyste (C9-8) : décisions CONFIRM / REJECT / ESCALATE, statut 'pending'
jusqu'à approbation, append-only, chaîné par hash, stocké hors des labels officiels.
Aucune décision ne modifie priority_final ni ne déclenche de réentraînement.
"""
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone

CONTRACT_VERSION = 'c9-4/v1'
ARTIFACTS_SUBDIR = os.path.join('reports', 'artifacts')
DECISIONS_SUBDIR = os.path.join('feedback', 'analyst_decisions')

# Niveau A commun à tous les artefacts
LEVEL_A_FIELDS = ('schema_version', 'artifact_id', 'kind', 'run_id', 'created_at',
                  'producer_cell', 'model_revision', 'adapter_sha256', 'scoring_version',
                  'payload_file', 'payload_sha256', 'payload_bytes', 'contract_version')

# Registre des types d'artefacts : schéma connu, nature du payload, champs exigés.
ARTIFACT_SCHEMAS = {
    'p2_shap': {
        'schema_version': '1', 'payload': 'json', 'level_b': True,
        'meta': ('explained_model_sha', 'feature_list_hash', 'scored_findings_sha256'),
        'record': ('finding_id', 'top_drivers', 'risk_score', 'status'),
    },
    'p3_propagation': {
        'schema_version': '1', 'payload': 'json', 'level_b': True,
        'meta': ('graph_hash', 'gnn_model_sha', 'params', 'scored_findings_sha256'),
        'record': ('finding_id', 'status', 'affected_assets', 'propagation_score'),
    },
    'p4_fp': {
        # v2 (08/11/2026) : marquage des findings deja labellises (probabilite non
        # indicative) et avertissements d'evaluation obligatoires.
        'schema_version': '2', 'payload': 'json', 'level_b': True,
        'meta': ('fp_model_sha', 'label_snapshot_hash', 'threshold', 'scored_findings_sha256',
                 'evaluation_caveats', 'auc_statistically_reliable'),
        'record': ('finding_id', 'fp_probability', 'routed_to_review', 'review_status',
                   'in_training_labels', 'probability_indicative'),
    },
    'p5_index': {
        'schema_version': '1', 'payload': 'binary', 'level_b': False,
        'meta': ('embedding_model', 'embedding_revision', 'embedding_model_sha',
                 'index_type', 'n_vectors', 'dim', 'id_order_sha256'),
        'record': (),
    },
    'p5_neighbors': {
        'schema_version': '1', 'payload': 'json', 'level_b': True,
        'meta': ('embedding_model', 'embedding_revision', 'embedding_model_sha',
                 'index_sha', 'top_k', 'scored_findings_sha256'),
        'record': ('finding_id', 'neighbors', 'label'),
    },
}

DECISION_ACTIONS = ('CONFIRM', 'REJECT', 'ESCALATE')
DECISION_MODULES = ('P2', 'P3', 'P4', 'P5')
# Fichiers officiels gelés : le journal ne doit JAMAIS y écrire.
OFFICIAL_LABEL_FILES = ('approved_label_snapshot.csv', 'analyst_labels.csv', 'fp_labels.csv')


class ArtifactContractError(Exception):
    """Artefact absent, altéré, de schéma inconnu ou non conforme au contrat."""


class DecisionRefused(Exception):
    """Décision analyste refusée (action, module, auteur ou destination invalide)."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def artifacts_dir(base_dir):
    return os.path.join(base_dir, ARTIFACTS_SUBDIR)


def _atomic_write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.tmp_')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1,
                      allow_nan=False, default=str).encode('utf-8')


def _validate_level_b(kind, payload):
    spec = ARTIFACT_SCHEMAS[kind]
    recs = payload.get('records') if isinstance(payload, dict) else None
    if not isinstance(recs, list):
        raise ArtifactContractError(f'[{kind}] payload sans liste "records" (niveau B)')
    for i, r in enumerate(recs):
        if not isinstance(r, dict) or not str(r.get('finding_id') or '').strip():
            raise ArtifactContractError(f'[{kind}] enregistrement {i} sans finding_id (niveau B)')
        missing = [f for f in spec['record'] if f not in r]
        if missing:
            raise ArtifactContractError(f'[{kind}] enregistrement {i} : champs manquants {missing}')


def write_artifact(base_dir, kind, payload, meta, *, run_id, producer_cell,
                   model_revision=None, adapter_sha256=None, scoring_version=None):
    """Écrit le payload puis son manifeste (niveau A). Rend le manifeste.

    payload : dict (JSON) ou bytes / chemin de fichier (binaire, ex. index FAISS).
    """
    if kind not in ARTIFACT_SCHEMAS:
        raise ArtifactContractError(f'type d artefact inconnu : {kind!r}')
    spec = ARTIFACT_SCHEMAS[kind]
    missing = [m for m in spec['meta'] if m not in (meta or {})]
    if missing:
        raise ArtifactContractError(f'[{kind}] métadonnées manquantes : {missing}')
    if spec['payload'] == 'json':
        if not isinstance(payload, dict):
            raise ArtifactContractError(f'[{kind}] payload JSON attendu')
        if spec['level_b']:
            _validate_level_b(kind, payload)
        data = _canonical_json_bytes(payload)
        payload_file = f'{kind}.json'
    else:
        if isinstance(payload, (bytes, bytearray)):
            data = bytes(payload)
        elif isinstance(payload, str) and os.path.isfile(payload):
            with open(payload, 'rb') as f:
                data = f.read()
        else:
            raise ArtifactContractError(f'[{kind}] payload binaire attendu (bytes ou chemin)')
        payload_file = f'{kind}.bin'
    d = artifacts_dir(base_dir)
    manifest = {
        'schema_version': spec['schema_version'],
        'artifact_id': f'{kind}-{uuid.uuid4().hex[:12]}',
        'kind': kind,
        'run_id': run_id,
        'created_at': _now(),
        'producer_cell': producer_cell,
        'model_revision': model_revision,
        'adapter_sha256': adapter_sha256,
        'scoring_version': scoring_version or {},
        'payload_file': payload_file,
        'payload_sha256': sha256_bytes(data),
        'payload_bytes': len(data),
        'contract_version': CONTRACT_VERSION,
        'meta': dict(meta),
    }
    _atomic_write_bytes(os.path.join(d, payload_file), data)
    _atomic_write_bytes(os.path.join(d, f'{kind}.manifest.json'), _canonical_json_bytes(manifest))
    return manifest


def read_artifact(base_dir, kind):
    """Relit un artefact en vérifiant le contrat. Rend (manifest, payload).

    payload : dict pour un artefact JSON, chemin du fichier vérifié pour un binaire.
    Lève ArtifactContractError si absent, schéma inconnu ou hash différent.
    """
    if kind not in ARTIFACT_SCHEMAS:
        raise ArtifactContractError(f'type d artefact inconnu : {kind!r}')
    spec = ARTIFACT_SCHEMAS[kind]
    d = artifacts_dir(base_dir)
    mp = os.path.join(d, f'{kind}.manifest.json')
    if not os.path.isfile(mp):
        raise ArtifactContractError(f'[{kind}] manifeste absent : {mp}')
    with open(mp, encoding='utf-8') as f:
        manifest = json.load(f)
    missing = [k for k in LEVEL_A_FIELDS if k not in manifest]
    if missing:
        raise ArtifactContractError(f'[{kind}] manifeste incomplet (niveau A) : {missing}')
    if manifest.get('kind') != kind:
        raise ArtifactContractError(f'[{kind}] manifeste d un autre type : {manifest.get("kind")!r}')
    if str(manifest.get('schema_version')) != spec['schema_version']:
        raise ArtifactContractError(f'[{kind}] schema_version inconnu : {manifest.get("schema_version")!r}')
    pp = os.path.join(d, os.path.basename(manifest['payload_file']))
    if not os.path.isfile(pp):
        raise ArtifactContractError(f'[{kind}] payload absent : {pp}')
    if sha256_file(pp) != manifest['payload_sha256']:
        raise ArtifactContractError(f'[{kind}] payload altéré : SHA-256 différent du manifeste')
    if spec['payload'] == 'binary':
        return manifest, pp
    with open(pp, encoding='utf-8') as f:
        payload = json.load(f)
    if spec['level_b']:
        _validate_level_b(kind, payload)
    return manifest, payload


def load_artifact_safe(base_dir, kind):
    """Pour l'application : rend (manifest, payload, None) ou (None, None, raison).
    Ne lève jamais : un module consultatif indisponible ne casse rien (C9-6)."""
    try:
        m, p = read_artifact(base_dir, kind)
        return m, p, None
    except Exception as e:                      # noqa: BLE001 — affiché tel quel
        return None, None, f'{type(e).__name__}: {e}'


# ─────────────────────────────────────────────────────────────────────────────
# Journal des décisions analyste (C9-8)
# ─────────────────────────────────────────────────────────────────────────────
def decisions_path(base_dir):
    return os.path.join(base_dir, DECISIONS_SUBDIR, 'decisions.jsonl')


def _check_decision_path(base_dir, path):
    root = os.path.realpath(os.path.join(base_dir, DECISIONS_SUBDIR))
    rp = os.path.realpath(path)
    if not rp.startswith(root + os.sep):
        raise DecisionRefused(f'écriture refusée hors de {root}')
    if os.path.basename(rp) in OFFICIAL_LABEL_FILES:
        raise DecisionRefused('écriture refusée dans un fichier de labels officiels')


def read_analyst_decisions(base_dir):
    p = decisions_path(base_dir)
    if not os.path.isfile(p):
        return []
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def verify_decision_journal(base_dir):
    """Vérifie le chaînage (prev_hash) du journal. Rend (ok, index_rupture)."""
    prev = '0' * 64
    for i, d in enumerate(read_analyst_decisions(base_dir)):
        body = {k: v for k, v in d.items() if k != 'entry_hash'}
        if d.get('prev_hash') != prev or sha256_bytes(_canonical_json_bytes(body)) != d.get('entry_hash'):
            return False, i
        prev = d['entry_hash']
    return True, None


def record_analyst_decision(base_dir, finding_id, module, action, reviewer, comment='',
                            artifact_id=None):
    """Ajoute UNE décision au journal (append-only, statut 'pending').

    N'écrit jamais dans les labels officiels, ne modifie jamais priority_final,
    ne déclenche aucun réentraînement : l'approbation reste une étape séparée.
    """
    action = str(action or '').upper().strip()
    module = str(module or '').upper().strip()
    if action not in DECISION_ACTIONS:
        raise DecisionRefused(f'action invalide : {action!r} (attendu {DECISION_ACTIONS})')
    if module not in DECISION_MODULES:
        raise DecisionRefused(f'module invalide : {module!r} (attendu {DECISION_MODULES})')
    if not str(finding_id or '').strip():
        raise DecisionRefused('finding_id obligatoire')
    if not str(reviewer or '').strip():
        raise DecisionRefused('auteur (reviewer) obligatoire')
    p = decisions_path(base_dir)
    _check_decision_path(base_dir, p)
    ok, broken = verify_decision_journal(base_dir)
    if not ok:
        raise DecisionRefused(f'journal altéré (rupture à l entrée {broken}) : écriture bloquée')
    prior = read_analyst_decisions(base_dir)
    entry = {
        'decision_id': uuid.uuid4().hex,
        'finding_id': str(finding_id).strip(),
        'module': module,
        'action': action,
        'status': 'pending',                 # jusqu'à approbation séparée
        'reviewer': str(reviewer).strip(),
        'comment': str(comment or '')[:2000],
        'artifact_id': artifact_id,
        'affects_priority_final': False,
        'triggers_retraining': False,
        'created_at': _now(),
        'prev_hash': prior[-1]['entry_hash'] if prior else '0' * 64,
    }
    entry['entry_hash'] = sha256_bytes(_canonical_json_bytes(entry))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    return entry
