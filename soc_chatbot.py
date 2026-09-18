import os, json, re, unicodedata
import pandas as pd
import streamlit as st

# ---- prompt systeme versionne (cf. prompt_version dans CFG) ----------------
CHATBOT_PROMPT_VERSION = "soc-chat-v6"
SYSTEM_PROMPT = (
    "Tu es un assistant SOC a DEUX MODES DE REPONSE, marques explicitement. "
    "MODE [CONTEXTE] : la question porte sur un finding, un chiffre ou un fait "
    "de ce projet, et l'information EST dans le CONTEXTE ou le GLOSSAIRE fournis. "
    "Commence ta reponse par le tag exact '[CONTEXTE]' puis reponds UNIQUEMENT a "
    "partir de ce qui est fourni. Cite les identifiants de findings sur lesquels "
    "tu t'appuies. Ne calcule AUCUNE statistique et n'invente aucun chiffre : les "
    "valeurs presentes dans le contexte font foi. Ne propose pas de CVE, d'URL ou "
    "de version qui ne sont pas dans le contexte.\n"
    # v7 — mode connaissance generale, ajoute a la demande de l auteure : le
    # mode [CONTEXTE] seul renvoyait 'information absente' des qu une question
    # sortait du sous-ensemble recupere, meme pour une question de culture
    # generale en cybersecurite que le modele sait repondre sans halluciner de
    # DONNEES DE CE PROJET. Le risque d hallucination visait des faits projet
    # (un CVE, un chiffre, une version) inventes -- pas la connaissance
    # generale du domaine, qui reste utile si elle est etiquetee comme telle.
    "MODE [CONNAISSANCE GENERALE] : la question NE porte PAS sur un fait "
    "specifique a ce projet (pas de finding, pas de chiffre, pas de donnee de ce "
    "dataset), OU porte sur un fait specifique qui EST ABSENT du CONTEXTE et du "
    "GLOSSAIRE. Dans ce cas commence ta reponse par le tag exact "
    "'[CONNAISSANCE GENERALE]', PUIS reponds avec tes connaissances generales en "
    "cybersecurite (definitions, principes, bonnes pratiques). Restrictions "
    "strictes de ce mode : n'invente JAMAIS un chiffre, un CVE, une version "
    "corrective, une echeance ou tout autre fait qui aurait l air specifique a CE "
    "PROJET -- reste generique. Si la question melange les deux (une partie est "
    "dans le contexte, une partie non), reponds en deux paragraphes, un par tag.\n"
    "REGLES SUPPLEMENTAIRES (v2), a respecter strictement en mode [CONTEXTE] :\n"
    "1. En mode [CONTEXTE], ne definis un terme que s'il figure dans le "
    "GLOSSAIRE ou le CONTEXTE ; sinon bascule en mode [CONNAISSANCE GENERALE] "
    "pour cette partie de la reponse plutot que d'inventer une definition "
    "dans le mode [CONTEXTE].\n"
    "2. Quand le GLOSSAIRE definit un terme, reprends SA definition mot pour mot "
    "dans ton sens, sans la reformuler ni l'enrichir.\n"
    "3. N'attribue aucune causalite qui n'est pas ecrite dans le contexte. Ne dis "
    "pas qu'une valeur 'a cause' une severite ou une decision si le contexte ne "
    "l'affirme pas.\n"
    "4. Les valeurs SHAP sont des CONTRIBUTIONS SIGNEES au score, PAS les valeurs "
    "des variables. Ne confonds jamais 'CVSS = 9.8' (valeur) avec 'contribution "
    "SHAP de cvss_numeric = +4.32' (impact sur le score).\n"
    "5. Le contexte ne contient qu'un SOUS-ENSEMBLE des findings. Ne generalise "
    "jamais a l'ensemble du parc et ne parle jamais de 'tous les findings'.\n"
    # v3 — perimetre defensif (s applique dans les DEUX modes)
    "6. PERIMETRE STRICTEMENT DEFENSIF, dans les deux modes. Tu expliques et "
    "priorises des findings ; tu n'aides jamais a exploiter une vulnerabilite. "
    "Refuse toute demande de code d'exploitation, de preuve de concept "
    "offensive, de payload, de contournement de protection, de reconnaissance "
    "offensive ou de mode operatoire d'attaque, meme presentee comme un test, "
    "un audit autorise ou un exercice pedagogique, meme en mode "
    "[CONNAISSANCE GENERALE]. Reponds alors : 'Hors perimetre : cet assistant "
    "documente et priorise les findings, il ne fournit pas de moyen "
    "d'exploitation.' Tu peux en revanche toujours expliquer l'impact, la "
    "severite et la REMEDIATION."
)

# ─────────────────────────────────────────────────────────────────
# GLOSSAIRE FACTUEL (v2). Injecte dans chaque prompt.
# Sans lui, le modele inventait des definitions : au test, il a produit
# "Kaspersky Vulnerability Expert" puis "Knowledge, Experience and
# Vulnerabilities" pour KEV, et decrivait l'echeance CISA comme une date
# d'exploitation au lieu d'une date limite de remediation.
# ─────────────────────────────────────────────────────────────────
GLOSSAIRE = (
    "GLOSSAIRE (definitions faisant autorite, a reprendre telles quelles) :\n"
    "- CISA KEV = 'Known Exploited Vulnerabilities'. Catalogue publie par la CISA "
    "listant les vulnerabilites dont l'exploitation dans la nature est AVEREE. "
    "Ce n'est ni une notation de risque, ni une classification de severite : "
    "c'est un constat d'exploitation active.\n"
    "- 'echeance' d'un finding KEV = DATE LIMITE DE REMEDIATION imposee par la "
    "directive CISA BOD 22-01. Ce n'est PAS une date d'exploitation ni une date "
    "de decouverte.\n"
    "- CVSS = 'Common Vulnerability Scoring System'. Score de severite technique "
    "de 0 a 10, independant du contexte d'exploitation reel.\n"
    "- EPSS = 'Exploit Prediction Scoring System'. Probabilite estimee (0 a 1) "
    "qu'une vulnerabilite soit exploitee dans les 30 jours.\n"
    "- SHAP = 'SHapley Additive exPlanations'. Methode d'explicabilite qui "
    "attribue a chaque variable une CONTRIBUTION SIGNEE au score predit. Une "
    "contribution positive augmente le score, une negative le diminue. La "
    "contribution SHAP d'une variable est distincte de la valeur de cette "
    "variable.\n"
    "- risk_score = score de risque de 0 a 100 produit par le modele ML de la "
    "plateforme (ensemble XGBoost/LightGBM), distinct du CVSS.\n"
)

def _base_dir():
    return os.environ.get("SOC_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))

def _secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

@st.cache_data(ttl=60)
def load_data():
    base = _base_dir()
    scored_path = os.path.join(base, "normalized_alerts", "scored_findings.csv")
    agents_path = os.path.join(base, "reports", "agent_results.json")
    df = pd.read_csv(scored_path) if os.path.exists(scored_path) else pd.DataFrame()
    if not df.empty and "id" in df.columns:
        df["id"] = df["id"].astype(str).str.strip()
    agents = []
    if os.path.exists(agents_path):
        try:
            with open(agents_path, encoding="utf-8") as f:
                agents = json.load(f)
        except Exception:
            agents = []
    return df, agents

def _fmt_finding(row, agent):
    # construit un bloc texte a partir des CHAMPS REELS presents seulement
    def g(k):
        v = row.get(k)
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        return v
    parts = [f"Finding ID: {g('id')}"]
    for label, key in [("Titre", "title"), ("Severite", "severity"),
                       ("Risk score", "risk_score"), ("CVSS", "cvss_score"),
                       ("EPSS", "epss_score"), ("Package", "package"),
                       ("Cible", "target"), ("Version corrective", "fix_version"),
                       ("Detecte par", "detected_by"),
                       ("Expose sur Internet", "internet_facing"),
                       ("Criticite actif", "asset_criticality")]:
        val = g(key)
        if val is not None and str(val) != "":
            parts.append(f"{label}: {val}")
    # KEV : n affirme 'exploite activement' que si le flag reel est vrai
    in_kev = g("in_kev")
    if str(in_kev).strip().lower() in ("1", "true", "yes"):
        due = g("kev_due_date")
        parts.append("Statut CISA KEV: dans le catalogue (exploite activement)"
                     + (f", echeance {due}" if due else ""))
    shap = g("top_shap_drivers")
    if shap is not None and str(shap) != "":
        # v2: etiquetage explicite. Sans lui, le modele presentait les VALEURS
        # des variables (CVSS=9.8) comme si c'etaient des contributions SHAP.
        parts.append("Contributions SHAP au risk_score (valeurs signees : "
                     "positif = augmente le score, negatif = le diminue ; "
                     "a ne pas confondre avec la valeur de la variable): "
                     f"{shap}")
    if isinstance(agent, dict):
        # NV Cas A : la priorite affichee vient du scorer ML (priority_final),
        # jamais de triage.priority du LLM. Le LLM ne fournit plus que
        # l explication (triage.reason), la conformite, la remediation et la
        # strategie de risque. priority_llm_legacy est conserve dans les
        # donnees pour audit mais n est PAS expose au chatbot.
        _pf = agent.get("priority_final")
        if _pf:
            parts.append(f"Priorite (scorer ML, risk_score_oof): {_pf}")
        tri = agent.get("triage", {})
        if isinstance(tri, dict) and tri.get("reason"):
            parts.append(f"Explication du triage (LLM): {tri.get('reason')}")
        comp = agent.get("compliance", {})
        if isinstance(comp, dict) and comp.get("controls"):
            parts.append(f"Controles de conformite: {comp.get('controls')}")
        rem = agent.get("remediation", {})
        if isinstance(rem, dict) and rem.get("patch_command"):
            parts.append(f"Commande de remediation: {rem.get('patch_command')}")
        rt = agent.get("red_team_challenge", {})
        if isinstance(rt, dict):
            if rt.get("final_verdict"):
                parts.append(f"Verdict red-team: {rt.get('final_verdict')}")
            if rt.get("challenge_summary"):
                parts.append(f"Synthese red-team: {rt.get('challenge_summary')}")
            if rt.get("false_positive_risk"):
                parts.append(f"Risque faux positif red-team: {rt.get('false_positive_risk')}")
            changes = rt.get("recommended_changes")
            if changes:
                parts.append(f"Changements recommandes red-team: {changes}")
    return "\n".join(parts)

@st.cache_data(ttl=60)
def build_context(_df, _agents):
    by_id = {}
    agent_by_id = {}
    for a in (_agents or []):
        if isinstance(a, dict):
            fid = str(a.get("finding_id", "")).strip()
            if fid and fid not in agent_by_id:
                agent_by_id[fid] = a
    seen = {}
    for _, row in _df.iterrows():
        d = row.to_dict()
        base_fid = str(d.get("id", "")).strip() or f"row-{len(by_id) + 1}"
        seen[base_fid] = seen.get(base_fid, 0) + 1
        fid = base_fid if seen[base_fid] == 1 else f"{base_fid}#{seen[base_fid]}"
        by_id[fid] = _fmt_finding(d, agent_by_id.get(base_fid, {}))
    return by_id

def _context_base_id(fid):
    m = re.match(r"^(.*)#\d+$", str(fid))
    return m.group(1) if m else str(fid)

def retrieve(question, context_by_id, k=4):
    q = _norm(question)          # v6: accents neutralises cote retrieval
    # 1) match exact d un identifiant present dans la question, en preservant
    # les lignes dupliquees conservees sous forme ID#2, ID#3, ...
    exact = [(fid, txt) for fid, txt in context_by_id.items()
             if _context_base_id(fid) and _context_base_id(fid).lower() in q]
    if exact:
        return exact[:k]
    # 2) similarite lexicale simple (recouvrement de mots) — pas d embeddings ici,
    #    volontairement transparent et sans dependance lourde
    qwords = set(w for w in q.replace(",", " ").split() if len(w) > 2)
    scored = []
    for fid, txt in context_by_id.items():
        tw = set(txt.lower().split())
        overlap = len(qwords & tw)
        if overlap:
            scored.append((overlap, fid, txt))
    scored.sort(reverse=True)
    hits = [(fid, txt) for _, fid, txt in scored[:k]]
    # NV61 smoke-test: si le recouvrement lexical est nul (question dans une
    # autre langue que le contexte, ou termes generaux), on ne renvoie JAMAIS
    # une liste vide -- on retombe sur les findings au plus haut risque, pour
    # que le LLM ait toujours un contexte reel a citer plutot que rien.
    if not hits:
        def _rk(item):
            _txt = item[1]
            import re as _re
            _m = _re.search(r"Risk score: ([0-9.]+)", _txt)
            return float(_m.group(1)) if _m else 0.0
        hits = sorted(context_by_id.items(), key=_rk, reverse=True)[:k]
    return hits

# NV86 — la liste statique de modeles s est perimee en production : les 5
# candidats (y compris gemma2-9b-it, le dernier tente) etaient tous
# decommissionnes cote Groq ("model_decommissioned"), la synthese LLM
# tombait en panne totale malgre le retrieval qui, lui, fonctionnait. Une
# liste ecrite en dur pourrit forcement — Groq retire des modeles sans
# prevenir le code qui les appelle. On interroge desormais l API Groq
# elle-meme pour la liste des modeles REELLEMENT actifs a l instant du run,
# mise en cache 1h pour ne pas payer un appel supplementaire par question.
#
# NB: pas de docstring triple-quote ici — ce fichier entier est lui-meme le
# contenu d une chaine r-triple-quote (CHATBOT_CODE). Un triple-quote imbrique
# fermerait cette chaine prematurement et casserait soc_chatbot.py genere.
_FALLBACK_CANDIDATES = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]

@st.cache_data(ttl=3600, show_spinner=False)
def _discover_groq_models(api_key):
    # Modeles chat actifs selon Groq, plus recents/gros en tete.
    # Retourne [] si la decouverte echoue (repli sur _FALLBACK_CANDIDATES).
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        models = client.models.list().data
    except Exception:
        return []
    # exclure whisper (audio), guard/moderation (non conversationnels) et
    # tout modele que Groq marque lui-meme comme non actif quand ce champ existe.
    _EXCLUDE = ("whisper", "guard", "moderation", "tts")
    # NV96 — incident constate : gemma2-9b-it revenait dans la decouverte
    # AVEC active=True cote Groq, alors que l appel chat/completions renvoyait
    # 400 "model_decommissioned". Le champ 'active' de /models n est donc pas
    # une garantie de disponibilite reelle -- Groq peut retirer un modele du
    # service avant de mettre a jour ses metadonnees de listing. On maintient
    # en plus une liste noire cote client pour les modeles dont on SAIT,
    # par incident constate, qu ils sont retires malgre un statut 'active'
    # trompeur. A completer si un nouvel incident du meme type survient.
    _KNOWN_DECOMMISSIONED = ("gemma2-9b-it", "gemma-7b-it",
                             "mixtral-8x7b-32768", "llama2-70b-4096")
    names = [m.id for m in models
             if not any(x in m.id.lower() for x in _EXCLUDE)
             and m.id.lower() not in _KNOWN_DECOMMISSIONED
             and getattr(m, "active", True)]
    # NV92 — ordre INVERSE par rapport a NV86. Le tri placait les gros modeles
    # (versatile / 70b / 120b) en tete, donc le chatbot choisissait le plus LENT
    # pendant une soutenance. On privilegie la latence : les modeles "instant" /
    # petits d abord, les gros en repli. Pour forcer un modele precis, poser le
    # secret Streamlit 'chatbot_model' (il prime sur cet ordre).
    _SLOW = ("versatile", "70b", "120b", "-large")
    names.sort(key=lambda n: (1 if any(x in n.lower() for x in _SLOW) else 0, n))
    return names


def _preferred_models(api_key=None):
    # priorite : variable d env / secret 'chatbot_model' si l utilisateur l a
    # posee (ex. ecrite par la cellule PROBE), puis les modeles decouverts en
    # direct aupres de Groq, puis le repli statique si la decouverte echoue.
    forced = os.environ.get("chatbot_model", "") or _secret("chatbot_model", "")
    discovered = _discover_groq_models(api_key) if api_key else []
    base = discovered or _FALLBACK_CANDIDATES
    ordered = ([forced] if forced else []) + [m for m in base if m != forced]
    return ordered

def _is_model_error(err_text):
    # NV95: le fallback multi-modeles ne se declenchait QUE sur une erreur
    # "modele" (decommissionne/introuvable). Une limite de debit (rate limit)
    # ou un timeout reseau sur le PREMIER modele candidat faisait donc
    # renvoyer __ERROR__ immediatement, sans jamais essayer les modeles
    # suivants -- alors que sur Groq free-tier chaque modele a son propre
    # quota par minute : le modele B peut tres bien repondre quand A est
    # temporairement sature. C est le scenario le plus probable derriere un
    # "chatbot indisponible" pendant une demo (plusieurs questions rapprochees
    # epuisent le quota du modele instant en tete de liste).
    # On NE rajoute PAS "api key"/"quota"/"credit" ici : ces erreurs sont
    # liees a la cle, pas au modele -- elles echoueraient de la meme facon
    # sur tous les candidats, donc inutile d essayer les suivants.
    t = str(err_text).lower()
    return any(k in t for k in ("model", "not found", "decommission",
                                "does not exist", "deprecat",
                                "rate limit", "rate_limit", "429",
                                "timeout", "timed out", "connection", "503"))

# ─────────────────────────────────────────────────────────────────
# GARDE-FOU AGREGATION (v2)
# Au test, "quelle est la moyenne des risk scores de tous les findings ?"
# a produit un calcul (interdit par le prompt), sur les 4 findings recuperes
# seulement (pas les 156), ET arithmetiquement FAUX : le modele a annonce
# 38.25 la ou (25.5+33.9+58.5+58.5)/4 = 44.1.
# Un LLM ne doit pas faire d'arithmetique sur des donnees tabulaires quand
# le dataframe complet est disponible. On detecte la question agregative,
# on calcule en pandas sur l'INTEGRALITE des findings, et on renvoie le
# resultat sans passer par le modele.
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
# GARDE-FOU PERIMETRE (v3)
# Le prompt seul ne suffit pas : une consigne peut etre contournee par
# reformulation. On intercepte AVANT l'appel LLM les demandes offensives
# (exploit, PoC, payload, contournement). Le refus devient une propriete
# du code, pas une politesse du modele.
# Volontairement restreint aux demandes d'ACTION offensive : les mots
# "exploitation active", "exploitable" ou "exploite" employes pour DECRIRE
# un finding (statut KEV, EPSS) ne doivent PAS declencher le refus.
# ─────────────────────────────────────────────────────────────────
# v6 — NORMALISATION DES ACCENTS.
# L audit a montre que "ecris un exploit" etait bloque mais "écris un exploit"
# passait, et que "repartition"/"mediane" declenchaient le calcul pandas alors
# que "répartition"/"médiane" partaient au LLM. Les motifs sont ecrits sans
# accents ; on normalise donc TOUTE question avant analyse (NFD + suppression
# des diacritiques). Sans ca, il suffisait d ecrire un francais correct pour
# contourner les deux garde-fous.
def _norm(text):
    t = unicodedata.normalize("NFD", str(text or ""))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return t.replace("\u2019", "'").lower()


# v5 — FILTRE ELARGI. L audit a montre que la v3, qui exigeait un verbe
# precis colle au mot offensif, laissait passer 5 formulations sur 5 :
# "j ai besoin d un exploit", "rediger un poc", "fais moi un payload",
# "peux-tu m aider a exploiter X", "exploit code pour Y".
# Nouvelle logique en trois niveaux, pour bloquer large SANS bloquer le
# vocabulaire descriptif ("exploit connu", "exploitable", "exploite
# activement") qui est legitime et central dans un SOC.

# 1) Termes sans usage descriptif possible ici : bloques inconditionnellement.
_OFFENSIVE_NOUNS = re.compile(
    r"\b(payload|poc|preuve de concept|proof of concept|reverse shell|"
    r"shellcode|malware|backdoor|porte derobee|metasploit|exploit kit)\b",
    re.IGNORECASE)

# 2) "exploit" est ambigu (un finding PEUT avoir un exploit connu). On ne
#    bloque que s il y a une intention de DEMANDE ou de PRODUCTION.
_REQUEST_VERBS = (r"ecris|ecrire|redige|rediger|genere|generer|donne|donner|"
                  r"fournis|fournir|montre|montrer|fais|faire|besoin|aide|"
                  r"aider|veux|voudrais|peux|pourrais|write|create|generate|"
                  r"give|need|build|code|script")
_EXPLOIT_REQUEST = re.compile(
    r"(" + _REQUEST_VERBS + r")\b[^.?!]{0,80}\b(exploit|exploitation|payload|poc)"
    r"|\b(exploit|exploitation)\b[^.?!]{0,30}\b(code|script|payload|poc)\b"
    r"|\b(etape|etapes|step|steps)\b[^.?!]{0,60}\b(exploitation|exploiter|attack|attaque)\b"
    r"|\b(requete|request)\b[^.?!]{0,80}\b(declenche|trigger|triggers|causes?)\b"
    r"[^.?!]{0,80}\b(rce|deserialization|deserialisation|injection|exploit|exploitation)\b",
    re.IGNORECASE)

# 3) Verbes d action offensive a l infinitif, dans un contexte de demande.
_OFFENSIVE_ACTION = re.compile(
    r"(comment|how to|aide|aider|peux|pourrais|veux|voudrais|pour)"
    r"[^.?!]{0,40}\b(exploiter|attaquer|pirater|compromettre|contourner|"
    r"bypasser|exploit|attack|hack|bypass)\b"
    r"|\b(exploiter|attaquer|compromettre|pirater)\s+(ce|cette|le|la|les|mon|"
    r"notre|un|une)\b",
    re.IGNORECASE)

OUT_OF_SCOPE_MSG = (
    "Hors perimetre : cet assistant documente et priorise les findings, il ne "
    "fournit pas de moyen d'exploitation. Je peux en revanche detailler l'impact, "
    "la severite, le statut KEV/EPSS et la remediation recommandee."
)

# ─────────────────────────────────────────────────────────────────
# DIAGNOSTIC D ERREUR API (v4)
# Avant : toute panne affichait le meme texte opaque "erreur reseau/API".
# Impossible de distinguer un depassement de debit d une cle invalide ou
# d un contexte trop long — y compris en pleine demo. On traduit desormais
# le message brut de l API en cause lisible, en gardant le detail technique.
# ─────────────────────────────────────────────────────────────────
def explain_api_error(err_text):
    # v4.1: on normalise underscores/tirets -> espaces. Les API renvoient
    # tantot un message en clair ("Incorrect API key provided"), tantot le
    # seul code machine ("invalid_api_key"), et l audit a montre que la
    # forme courte n etait pas reconnue.
    t = str(err_text).lower().replace("_", " ").replace("-", " ")
    if "rate" in t and "limit" in t:
        return ("Limite de debit atteinte (trop de requetes rapprochees). "
                "Attends quelques secondes et repose la question.")
    if "quota" in t or "insufficient" in t or "credit" in t:
        return "Quota de l'API epuise pour la periode en cours."
    if "context length" in t or ("context" in t and ("window" in t or "too long" in t)):
        return ("Contexte trop long pour le modele : trop de findings "
                "recuperes d'un coup. Pose une question plus ciblee (un ID).")
    if ("api key" in t or "apikey" in t or "authentication" in t
            or "unauthorized" in t or "401" in t):
        return ("Cle API refusee. Verifie le secret 'chatbot' dans les "
                "parametres Streamlit Cloud.")
    if "model" in t and ("not found" in t or "decommission" in t
                         or "deprecat" in t or "does not exist" in t):
        return "Modele indisponible cote fournisseur (aucun modele de repli n'a repondu)."
    if "import groq" in t or "no module named" in t or "modulenotfound" in t:
        return ("Paquet 'groq' absent de l'environnement : ajoute 'groq' a "
                "requirements.txt du depot.")
    if "timeout" in t or "timed out" in t or "connection" in t:
        return "Delai depasse ou reseau indisponible. Reessaie."
    return "Cause non identifiee (voir le detail technique ci-dessous)."


def is_out_of_scope(question):
    q = _norm(question)          # v6: accents neutralises
    return bool(_OFFENSIVE_NOUNS.search(q)
                or _EXPLOIT_REQUEST.search(q)
                or _OFFENSIVE_ACTION.search(q))


_AGG_PATTERNS = re.compile(
    r"\b(moyenne|mediane|median|average|total|somme|combien|nombre de|"
    r"pourcentage|proportion|repartition|distribution|maximum|minimum|"
    r"le plus eleve|le plus haut|le plus bas|classement|top\s*\d+|"
    # v3: les demandes d ENUMERATION doivent aussi passer par pandas, sinon
    # le LLM repond sur les 4 findings recuperes et la liste est incomplete.
    # Si answer_aggregate ne sait pas traiter la question, elle renvoie None
    # et la question repart normalement vers le LLM.
    r"quel|quelle|quels|quelles|lesquels|liste|lister|enumere)\b",
    re.IGNORECASE)

def is_aggregate_question(question):
    return bool(_AGG_PATTERNS.search(_norm(question)))   # v6: accents neutralises

def answer_aggregate(question, df):
    # Repond aux questions agregatives par un calcul pandas sur TOUT le
    # dataframe. Retourne None si la question n'est pas couverte -> on laisse
    # alors le LLM repondre (le prompt lui interdit deja de calculer).
    q = _norm(question)          # v6: "severite"/"repartition" accentues
    if df is None or df.empty or "risk_score" not in df.columns:
        return None
    rs = pd.to_numeric(df["risk_score"], errors="coerce").dropna()
    n_total = len(df)
    lines = [f"Calcul effectue par le code (pandas) sur l'integralite des "
             f"{n_total} findings, pas par le LLM."]

    if re.search(r"moyenne|average", q) and len(rs):
        lines.append(f"- Moyenne des risk_score : {rs.mean():.2f}")
    if re.search(r"mediane|median", q) and len(rs):
        lines.append(f"- Mediane des risk_score : {rs.median():.2f}")
    if re.search(r"maximum|le plus eleve|le plus haut", q) and len(rs):
        lines.append(f"- Maximum des risk_score : {rs.max():.2f}")
    if re.search(r"minimum|le plus bas", q) and len(rs):
        lines.append(f"- Minimum des risk_score : {rs.min():.2f}")
    if re.search(r"combien|nombre de|total", q):
        lines.append(f"- Nombre total de findings : {n_total}")
        if "severity" in df.columns:
            vc = df["severity"].astype(str).str.upper().value_counts()
            lines.append("- Repartition par severite : "
                         + ", ".join(f"{k}={v}" for k, v in vc.items()))
        if "in_kev" in df.columns:
            _kev = df["in_kev"].astype(str).str.strip().str.lower().isin(
                ["1", "true", "yes", "1.0"]).sum()
            lines.append(f"- Findings dans le catalogue CISA KEV : {_kev}")
    if re.search(r"repartition|distribution", q) and "severity" in df.columns:
        vc = df["severity"].astype(str).str.upper().value_counts()
        lines.append("- Repartition par severite : "
                     + ", ".join(f"{k}={v}" for k, v in vc.items()))

    # v3 — ENUMERATION EXHAUSTIVE.
    # Le retrieval ne remonte que quelques findings : a la question "quels
    # findings sont dans le catalogue KEV ?", le LLM en listait 2 sur 3 —
    # reponse incomplete, indistinguable d'une erreur pour un lecteur.
    # Ces listes viennent donc du dataframe complet.
    if re.search(r"\b(quel|quelle|quels|quelles|liste|lister|lesquels|enumere)\b", q):
        if "kev" in q and "in_kev" in df.columns:
            _mask = df["in_kev"].astype(str).str.strip().str.lower().isin(
                ["1", "true", "yes", "1.0"])
            _ids = df.loc[_mask, "id"].astype(str).tolist()
            lines.append(f"- Findings dans le catalogue CISA KEV ({len(_ids)}) : "
                         + (", ".join(_ids) if _ids else "aucun"))
        for _sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            if _sev.lower() in q and "severity" in df.columns:
                _ids = df.loc[df["severity"].astype(str).str.upper() == _sev,
                              "id"].astype(str).tolist()
                _shown = ", ".join(_ids[:40]) + (" ..." if len(_ids) > 40 else "")
                lines.append(f"- Findings {_sev} ({len(_ids)}) : "
                             + (_shown if _ids else "aucun"))
                break

    return "\n".join(lines) if len(lines) > 1 else None


def ask_llm(question, retrieved):
    api_key = _secret("chatbot", os.environ.get("chatbot", ""))
    context = "\n\n---\n\n".join(txt for _, txt in retrieved) or "(aucun finding pertinent)"
    _n_ctx = len(retrieved)
    if not api_key:
        return None, context  # mode degrade signale par l appelant
    try:
        from groq import Groq
    except Exception as e:
        return f"__ERROR__import groq: {e}", context
    client = Groq(api_key=api_key)
    _last_err = None
    # NV96 — l ancienne version n exposait que le DERNIER echec dans le
    # message d erreur. L incident gemma2-9b-it a montre la limite : on ne
    # pouvait pas voir si un seul modele avait echoue ou si TOUS avaient
    # echoue pour des raisons differentes derriere ce dernier message. On
    # trace desormais chaque tentative (modele + cause courte), affiche dans
    # le meme expander technique deja present dans l UI (render_chatbot_tab).
    _attempts = []
    for _model in _preferred_models(api_key):
        try:
            resp = client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": (f"{GLOSSAIRE}\n"
                                 f"CONTEXTE ({_n_ctx} finding(s) recuperes sur "
                                 f"un ensemble plus large):\n{context}\n\n"
                                 f"QUESTION: {question}")},
                ],
                temperature=0.2, max_tokens=600,
            )
            # NV92: memoriser le modele ayant reellement repondu, pour que
            # l onglet puisse l afficher. Sans ca, une panne le jour J n est
            # pas diagnosticable depuis l interface.
            st.session_state["_soc_last_model"] = _model
            return resp.choices[0].message.content, context
        except Exception as e:
            _last_err = e
            _attempts.append(f"{_model}: {str(e)[:120]}")
            # si c est une erreur de modele, on essaie le candidat suivant ;
            # sinon (reseau, auth, quota) inutile d insister, on sort.
            if _is_model_error(e):
                continue
            _trace = " | ".join(_attempts)
            return f"__ERROR__{e}\n\n[trace des {len(_attempts)} tentative(s)] {_trace}", context
    _trace = " | ".join(_attempts) if _attempts else "aucune tentative (liste de candidats vide)"
    return (f"__ERROR__aucun modele Groq disponible parmi "
            f"{_preferred_models(api_key)} (dernier essai: {_last_err})\n\n"
            f"[trace des {len(_attempts)} tentative(s)] {_trace}"), context

# NV97 — a la demande de l auteure : si l info n est pas dans le CONTEXTE, le
# modele peut repondre avec ses connaissances generales plutot que refuser
# systematiquement (cf. SYSTEM_PROMPT, modes [CONTEXTE]/[CONNAISSANCE GENERALE]).
# Cote UI, les deux modes restent visuellement distincts : une reponse
# [CONNAISSANCE GENERALE] ne doit jamais avoir l air aussi "sourcee" qu une
# reponse [CONTEXTE], sinon on reintroduit le risque d hallucination que les
# regles v2 avaient ete ecrites pour eliminer -- on ne le supprime pas, on
# le rend visible.
_MODE_TAG_RE = re.compile(r"\[(CONTEXTE|CONNAISSANCE GENERALE)\]\s*", re.IGNORECASE)

def _render_tagged_answer(answer):
    if not isinstance(answer, str) or not _MODE_TAG_RE.search(answer):
        st.markdown(answer)
        return
    matches = list(_MODE_TAG_RE.finditer(answer))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(answer)
        mode = m.group(1).upper()
        txt = answer[start:end].strip()
        if not txt:
            continue
        if mode == "CONTEXTE":
            st.markdown(txt)
        else:
            st.info(" **Connaissance generale du modele** (non verifiee sur "
                    "les findings de ce projet) :\n\n" + txt)

def render_chatbot_tab():
    st.subheader("Assistant SOC — interroge tes findings")
    st.caption(f"RAG sur scored_findings.csv + agent_results.json "
               f"(prompt {CHATBOT_PROMPT_VERSION}). Reponse [CONTEXTE] = donnees "
               f"de ce projet uniquement. Reponse [CONNAISSANCE GENERALE] = culture "
               f"cybersecurite du modele, non verifiee sur ce dataset (affichee a part).")
    _lm = st.session_state.get("_soc_last_model")
    if _lm:
        st.caption(f"Modele LLM utilise : {_lm}")
    try:
        df, agents = load_data()
    except Exception as e:
        st.error(f"Artefacts illisibles: {e}")
        return
    if df.empty:
        st.warning("scored_findings.csv absent ou vide — lance le pipeline d'abord.")
        return
    context_by_id = build_context(df, agents)

    has_key = bool(_secret("chatbot", os.environ.get("chatbot", "")))
    if not has_key:
        st.info("Cle chatbot absente : mode degrade. Les findings pertinents "
                "sont affiches, mais la synthese LLM est desactivee.")

    suggestions = [
        "Pourquoi ce finding a-t-il un risk score eleve ?",
        "Quels findings sont dans le catalogue CISA KEV ?",
        "Quelle est la remediation recommandee et sa version corrective ?",
    ]
    cols = st.columns(len(suggestions))
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    _clicked = None
    for _c, _s in zip(cols, suggestions):
        if _c.button(_s, use_container_width=True):
            _clicked = _s

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_tagged_answer(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))

    user_q = st.chat_input("Pose ta question sur un finding...") or _clicked
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        retrieved = retrieve(user_q, context_by_id)
        sources = [fid for fid, _ in retrieved]
        # v2: les questions agregatives ne partent PAS au LLM — calcul pandas
        # sur l'integralite du dataframe, resultat exact et verifiable.
        # v3: perimetre defensif verifie EN PREMIER, avant tout appel LLM.
        if is_out_of_scope(user_q):
            _agg = OUT_OF_SCOPE_MSG
            sources = []
        else:
            _agg = answer_aggregate(user_q, df) if is_aggregate_question(user_q) else None
        if _agg is not None:
            answer, context = _agg, ""
            if _agg is not OUT_OF_SCOPE_MSG:
                sources = [f"calcul pandas sur {len(df)} findings"]
        else:
            answer, context = ask_llm(user_q, retrieved)
        with st.chat_message("assistant"):
            if answer is None:
                st.markdown("**Synthese LLM indisponible (pas de cle API).** "
                            "Findings pertinents retrouves :")
                for fid, txt in retrieved:
                    with st.expander(fid):
                        st.text(txt)
            elif isinstance(answer, str) and answer.startswith("__ERROR__"):
                # v4: on montre la CAUSE, plus un simple "indisponible".
                _raw = answer[len("__ERROR__"):]
                _cause = explain_api_error(_raw)
                st.warning(f"Synthese LLM indisponible — {_cause}")
                with st.expander("Detail technique de l'erreur"):
                    st.code(_raw or "(vide)")
                st.caption("Le retrieval a fonctionne : findings pertinents "
                           "ci-dessous, sans synthese LLM.")
                for fid, txt in retrieved:
                    with st.expander(fid):
                        st.text(txt)
                answer = f"(LLM indisponible — {_cause})"
            else:
                _render_tagged_answer(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer or "(mode degrade)",
             "sources": sources})

# Permet un lancement autonome (streamlit run soc_chatbot.py) pour test isole
if __name__ == "__main__":
    st.set_page_config(page_title="SOC Assistant", layout="wide")
    render_chatbot_tab()
