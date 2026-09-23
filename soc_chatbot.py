import os
import json
import re
import unicodedata
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st


CHATBOT_PROMPT_VERSION = "soc-chat-v8-sft-redteam"

SYSTEM_PROMPT = """
Tu es l'assistant SOC du projet SOC-Audit-v7.

Tu réponds avec deux modes :

[CONTEXTE]
Utilise uniquement les données réellement présentes dans le contexte fourni.
Cite les IDs de findings et les fichiers sources utilisés.
N'invente aucun chiffre, CVE, score, version ou décision.
Ne généralise jamais un résultat local à tous les findings si le contexte ne le permet pas.
Les résultats expérimentaux doivent rester présentés avec leurs limites.

[CONNAISSANCE GENERALE]
Si la question est générale et ne concerne pas les données du projet, réponds avec
des connaissances générales de cybersécurité. Ne présente jamais ces informations
comme des résultats mesurés dans ce projet.

RÈGLES :
- Le score CVSS est distinct du risk_score.
- Le risk_score est produit par le modèle ML.
- Les explications de triage viennent du modèle SFT.
- Le verdict red-team vient du Layer 7B.
- Le Layer 7B utilise le modèle de base, pas le SFT.
- P3 GNN est expérimental ; BFS reste actif.
- P4 est exploratoire avec peu de labels faux positifs.
- P7 est une expérience séparée et n'est pas promu.
- P6 (EWC) est implémenté mais n'a pas été exécuté dans ce projet. Il n'existe aucun résultat expérimental P6 à présenter.
- La recalibration OOF est mesurée par validation croisée.
- DPO n'est pas utilisé dans le pipeline principal.
- Ne fournis jamais de code d'exploitation, payload, PoC, reverse shell,
  procédure d'attaque ou méthode de contournement.
- Tu peux expliquer l'impact, la priorité, les preuves et la remédiation.
"""

GLOSSARY = """
GLOSSAIRE :
- CVSS : score technique de sévérité de 0 à 10.
- EPSS : probabilité estimée d'exploitation dans les 30 prochains jours.
- CISA KEV : catalogue des vulnérabilités dont l'exploitation est connue.
- risk_score : score de risque ML de la plateforme.
- SHAP : contribution signée d'une variable au score prédit.
- SFT : fine-tuning supervisé sur les labels humains approuvés.
- DPO : fine-tuning par préférences, utilisé uniquement dans l'annexe.
- OOF : prédiction obtenue sur un fold jamais utilisé pour l'entraînement.
- BFS : propagation déterministe du blast radius.
"""

OUT_OF_SCOPE_MESSAGE = (
    "Hors périmètre : cet assistant documente et priorise les findings. "
    "Il ne fournit pas de moyen d'exploitation. "
    "Je peux expliquer l'impact, la sévérité et la remédiation."
)


def _base_dir():
    return os.environ.get(
        "SOC_BASE_DIR",
        os.path.dirname(os.path.abspath(__file__)),
    )


def _secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default



CHAT_CACHE_TABLE = "llm_response_cache_chat"


def _chat_supabase_client():
    try:
        from supabase import create_client

        url = (
            st.secrets.get("SUPABASE_URL", "")
            or os.environ.get("SUPABASE_URL", "")
        )

        key = (
            st.secrets.get("SUPABASE_KEY", "")
            or os.environ.get("SUPABASE_KEY", "")
        )

        if not url or not key:
            return None

        return create_client(url, key)

    except Exception:
        return None


def _chat_cache_key(question, context, model_tag):
    raw = (
        str(question)
        + "\n"
        + str(context)
        + "\n"
        + CHATBOT_PROMPT_VERSION
        + "\n"
        + str(model_tag)
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def _chat_cache_get(cache_key):
    client = _chat_supabase_client()

    if client is None:
        return None

    try:
        rows = (
            client.table(CHAT_CACHE_TABLE)
            .select("payload")
            .eq("cache_key", cache_key)
            .limit(1)
            .execute()
            .data
        )

        if not rows:
            return None

        payload = rows[0].get("payload")

        if isinstance(payload, dict):
            return payload.get("answer")

        return payload

    except Exception:
        return None


def _chat_cache_put(cache_key, answer, model_name):
    client = _chat_supabase_client()

    if client is None or not answer:
        return

    try:
        client.table(CHAT_CACHE_TABLE).upsert(
            {
                "cache_key": cache_key,
                "payload": {
                    "answer": answer,
                    "prompt_version": CHATBOT_PROMPT_VERSION,
                },
                "model_name": model_name,
            }
        ).execute()

    except Exception:
        pass


def _norm(value):
    text = unicodedata.normalize("NFD", str(value or ""))
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    return text.lower().replace("’", "'").strip()


def _is_true(value):
    return _norm(value) in {"1", "1.0", "true", "yes", "oui"}


def _is_offensive(question):
    q = _norm(question)

    offensive_terms = [
        "payload",
        "proof of concept",
        "preuve de concept",
        "reverse shell",
        "shellcode",
        "malware",
        "backdoor",
        "metasploit",
        "exploit code",
        "code exploit",
        "ecris un exploit",
        "écris un exploit",
        "comment exploiter",
        "how to exploit",
        "comment attaquer",
        "how to attack",
        "contourner la protection",
        "bypass protection",
    ]

    return any(term in q for term in offensive_terms)


def _agent_list(value):
    if isinstance(value, list):
        return [
            item for item in value
            if isinstance(item, dict)
        ]

    if isinstance(value, dict):
        if isinstance(value.get("results"), list):
            return [
                item for item in value["results"]
                if isinstance(item, dict)
            ]

        output = []
        for finding_id, item in value.items():
            if isinstance(item, dict):
                row = dict(item)
                row.setdefault("finding_id", finding_id)
                output.append(row)
        return output

    return []


@st.cache_data(ttl=60, show_spinner=False)
def load_project_data():
    base = _base_dir()

    scored_path = os.path.join(
        base,
        "normalized_alerts",
        "scored_findings.csv",
    )

    agents_path = os.path.join(
        base,
        "reports",
        "agent_results_SFT_redteam.json",
    )

    df = (
        pd.read_csv(scored_path)
        if os.path.exists(scored_path)
        else pd.DataFrame()
    )

    if not df.empty and "id" in df.columns:
        df["id"] = df["id"].astype(str).str.strip()

    agents = []

    if os.path.exists(agents_path):
        try:
            with open(agents_path, encoding="utf-8") as handle:
                agents = _agent_list(json.load(handle))
        except Exception:
            agents = []

    report_names = [
        "system_card.md",
        "p3_gnn_bfs_comparison.json",
        "p4_fp_exploratory.json",
        "priority_to_score_calibre.json",
        "mae_oof_calibration_kaggle.json",
        "p7_debate_experiment.json",
        "p7_debate_bootstrap.json",
    ]

    reports = {}

    for name in report_names:
        path = os.path.join(base, "reports", name)

        if not os.path.exists(path):
            continue

        try:
            if name.endswith(".json"):
                with open(path, encoding="utf-8") as handle:
                    reports[name] = json.load(handle)
            else:
                with open(path, encoding="utf-8") as handle:
                    reports[name] = handle.read()
        except Exception:
            continue

    return df, agents, reports


def _clean_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none"}:
        return None

    return value


def _format_finding(row, agent):
    def get_value(key):
        return _clean_value(row.get(key))

    finding_id = get_value("id") or "unknown"

    parts = [
        f"Finding ID: {finding_id}",
    ]

    fields = [
        ("Title", "title"),
        ("Severity", "severity"),
        ("Risk score", "risk_score"),
        ("OOF risk score", "risk_score_oof"),
        ("Priority", "priority_final"),
        ("CVSS", "cvss_score"),
        ("EPSS", "epss_score"),
        ("Package", "package"),
        ("Target", "target"),
        ("Asset", "asset_id"),
        ("Fix version", "fix_version"),
        ("Detected by", "detected_by"),
        ("Internet facing", "internet_facing"),
        ("Asset criticality", "asset_criticality"),
    ]

    for label, key in fields:
        value = get_value(key)
        if value is not None:
            parts.append(f"{label}: {value}")

    if _is_true(get_value("in_kev")):
        due_date = get_value("kev_due_date")
        text = "CISA KEV: yes"
        if due_date is not None:
            text += f" | remediation due date: {due_date}"
        parts.append(text)

    shap = get_value("top_shap_drivers")
    if shap is not None:
        parts.append(
            "SHAP signed contributions to risk_score: "
            f"{shap}"
        )

    if isinstance(agent, dict):
        triage = agent.get("triage") or {}
        compliance = agent.get("compliance") or {}
        remediation = agent.get("remediation") or {}
        red_team = agent.get("red_team_challenge") or {}

        if isinstance(triage, dict) and triage.get("reason"):
            parts.append(
                f"LLM triage explanation: {triage['reason']}"
            )

        if isinstance(compliance, dict) and compliance.get("controls"):
            parts.append(
                f"Compliance controls: {compliance['controls']}"
            )

        if isinstance(remediation, dict):
            if remediation.get("patch_command"):
                parts.append(
                    "Recommended remediation command: "
                    f"{remediation['patch_command']}"
                )

            if remediation.get("verification"):
                parts.append(
                    f"Remediation verification: "
                    f"{remediation['verification']}"
                )

        if isinstance(red_team, dict):
            if red_team.get("final_verdict"):
                parts.append(
                    f"Red-team final verdict: "
                    f"{red_team['final_verdict']}"
                )

            if red_team.get("pipeline_status"):
                parts.append(
                    f"Red-team pipeline status: "
                    f"{red_team['pipeline_status']}"
                )

            if red_team.get("challenge_summary"):
                parts.append(
                    f"Red-team summary: "
                    f"{red_team['challenge_summary']}"
                )

            if red_team.get("recommended_changes"):
                parts.append(
                    f"Red-team recommended changes: "
                    f"{red_team['recommended_changes']}"
                )

    return "\n".join(parts)


def build_finding_index(df, agents):
    agent_by_id = {}

    for agent in agents:
        finding_id = str(
            agent.get("finding_id")
            or agent.get("id")
            or ""
        ).strip()

        if finding_id:
            agent_by_id[finding_id] = agent

    index = {}

    for _, row in df.iterrows():
        data = row.to_dict()
        finding_id = str(data.get("id", "")).strip()

        if not finding_id:
            continue

        index[finding_id] = _format_finding(
            data,
            agent_by_id.get(finding_id, {}),
        )

    return index


def _report_text(name, value):
    if isinstance(value, str):
        return f"REPORT {name}:\n{value[:12000]}"

    try:
        return (
            f"REPORT {name}:\n"
            f"{json.dumps(value, ensure_ascii=False, indent=2)[:12000]}"
        )
    except Exception:
        return f"REPORT {name}:\n{str(value)[:12000]}"


def retrieve(question, finding_index, reports, top_k=6):
    question_norm = _norm(question)

    # Exact finding ID first.
    exact = []

    for finding_id, text in finding_index.items():
        if _norm(finding_id) in question_norm:
            exact.append((finding_id, text))

    if exact:
        return exact[:top_k]

    question_words = {
        word
        for word in re.findall(r"[a-z0-9_.#-]+", question_norm)
        if len(word) >= 3
    }

    scored = []

    for finding_id, text in finding_index.items():
        text_words = set(
            re.findall(r"[a-z0-9_.#-]+", _norm(text))
        )

        overlap = len(question_words & text_words)

        if overlap:
            scored.append((overlap, finding_id, text))

    scored.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    results = [
        (finding_id, text)
        for _, finding_id, text in scored[:top_k]
    ]

    # Add relevant complementary reports.
    report_terms = {
        "p2": ["risk_explanations.json"],
        "shap": ["risk_explanations.json"],
        "p3": ["p3_gnn_bfs_comparison.json"],
        "gnn": ["p3_gnn_bfs_comparison.json"],
        "bfs": ["p3_gnn_bfs_comparison.json"],
        "p4": ["p4_fp_exploratory.json"],
        "faux positif": ["p4_fp_exploratory.json"],
        "false positive": ["p4_fp_exploratory.json"],
        "recalibration": [
            "priority_to_score_calibre.json",
            "mae_oof_calibration_kaggle.json",
        ],
        "mae": ["mae_oof_calibration_kaggle.json"],
        "p7": [
            "p7_debate_experiment.json",
            "p7_debate_bootstrap.json",
        ],
        "débat": [
            "p7_debate_experiment.json",
            "p7_debate_bootstrap.json",
        ],
        "dpo": ["system_card.md"],
        "p6": ["system_card.md"],
        "ewc": ["system_card.md"],
        "continual learning": ["system_card.md"],
        "sft": ["system_card.md"],
        "limite": ["system_card.md"],
    }

    selected_reports = []

    for term, names in report_terms.items():
        if term in question_norm:
            selected_reports.extend(names)

    for name in dict.fromkeys(selected_reports):
        if name in reports:
            results.append(
                (
                    f"REPORT:{name}",
                    _report_text(name, reports[name]),
                )
            )

    return results[:top_k + 3]


def _aggregate(question, df):
    if df is None or df.empty:
        return None

    q = _norm(question)
    lines = []

    if "risk_score" in df.columns:
        risk = pd.to_numeric(
            df["risk_score"],
            errors="coerce",
        ).dropna()

        if "moyenne" in q or "average" in q:
            lines.append(
                f"Moyenne risk_score : {risk.mean():.2f}"
            )

        if "mediane" in q or "median" in q:
            lines.append(
                f"Médiane risk_score : {risk.median():.2f}"
            )

        if "maximum" in q or "plus eleve" in q:
            lines.append(
                f"Maximum risk_score : {risk.max():.2f}"
            )

        if "minimum" in q or "plus bas" in q:
            lines.append(
                f"Minimum risk_score : {risk.min():.2f}"
            )

    if any(
        term in q
        for term in [
            "combien",
            "nombre",
            "total",
            "repartition",
            "distribution",
            "liste",
            "quels",
            "quelles",
            "lesquels",
        ]
    ):
        lines.append(f"Nombre total de findings : {len(df)}")

    if "severity" in df.columns:
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if severity.lower() in q:
                selected = df[
                    df["severity"].astype(str).str.upper()
                    == severity
                ]

                ids = (
                    selected["id"].astype(str).tolist()
                    if "id" in selected.columns
                    else []
                )

                lines.append(
                    f"Findings {severity} ({len(ids)}) : "
                    + ", ".join(ids[:50])
                )

    if "kev" in q and "in_kev" in df.columns:
        mask = df["in_kev"].map(_is_true)
        ids = (
            df.loc[mask, "id"].astype(str).tolist()
            if "id" in df.columns
            else []
        )

        lines.append(
            f"Findings CISA KEV ({len(ids)}) : "
            + (", ".join(ids) if ids else "aucun")
        )

    if not lines:
        return None

    return (
        "Calcul effectué par le code sur le dataframe complet. "
        "Le LLM n'a pas calculé ces valeurs.\n"
        + "\n".join(f"- {line}" for line in lines)
    )


def _model_candidates(api_key):
    forced = (
        os.environ.get("chatbot_model")
        or _secret("chatbot_model", "")
    )

    fallback = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
    ]

    discovered = []

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        for model in client.models.list().data:
            name = str(model.id)

            if any(
                term in name.lower()
                for term in [
                    "whisper",
                    "guard",
                    "moderation",
                    "tts",
                ]
            ):
                continue

            discovered.append(name)
    except Exception:
        pass

    ordered = [forced] if forced else []

    for model in discovered + fallback:
        if model and model not in ordered:
            ordered.append(model)

    return ordered


def ask_llm(question, retrieved, history):
    api_key = _secret(
        "chatbot",
        os.environ.get("chatbot", ""),
    )

    if not api_key:
        return None, "Clé Groq absente."

    try:
        from groq import Groq
    except Exception as error:
        return None, f"Module groq absent : {error}"

    context = "\n\n---\n\n".join(
        f"SOURCE {source}:\n{text}"
        for source, text in retrieved
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\n" + GLOSSARY,
        },
    ]

    for item in history[-6:]:
        if item["role"] in {"user", "assistant"}:
            messages.append(
                {
                    "role": item["role"],
                    "content": item["content"],
                }
            )

    messages.append(
        {
            "role": "user",
            "content": (
                f"CONTEXTE RÉCUPÉRÉ :\n{context}\n\n"
                f"QUESTION : {question}\n\n"
                "Réponds avec le tag [CONTEXTE] ou "
                "[CONNAISSANCE GENERALE]. "
                "Termine par une ligne Sources."
            ),
        }
    )

    client = Groq(api_key=api_key)
    errors = []

    for model in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.15,
                max_tokens=700,
            )

            st.session_state["_soc_last_model"] = model

            return response.choices[0].message.content, None

        except Exception as error:
            errors.append(f"{model}: {str(error)[:180]}")

    return None, " | ".join(errors)


def _render_answer(answer):
    if not answer:
        return

    st.markdown(answer)


def render_chatbot_tab():
    st.subheader("Assistant SOC — SFT + red-team")

    st.caption(
        "RAG sur scored_findings.csv, "
        "agent_results_SFT_redteam.json et les rapports "
        "P2/P3/P4/P7."
    )

    last_model = st.session_state.get("_soc_last_model")

    if last_model:
        st.caption(f"Modèle Groq utilisé : {last_model}")

    try:
        df, agents, reports = load_project_data()
    except Exception as error:
        st.error(f"Erreur de chargement des données : {error}")
        return

    if df.empty:
        st.warning(
            "scored_findings.csv absent ou vide."
        )
        return

    finding_index = build_finding_index(df, agents)

    if "soc_chat_history" not in st.session_state:
        st.session_state["soc_chat_history"] = []

    suggestions = [
        "Explique CVE-2022-22965",
        "Quels findings sont dans le catalogue KEV ?",
        "Pourquoi P3 n'a pas été promu ?",
        "Compare P7 au pipeline principal",
    ]

    columns = st.columns(len(suggestions))
    clicked = None

    for column, suggestion in zip(columns, suggestions):
        if column.button(
            suggestion,
            use_container_width=True,
        ):
            clicked = suggestion

    for message in st.session_state["soc_chat_history"]:
        with st.chat_message(message["role"]):
            _render_answer(message["content"])

            if message.get("sources"):
                st.caption(
                    "Sources : "
                    + ", ".join(message["sources"])
                )

    question = (
        st.chat_input("Pose ta question sur le projet SOC...")
        or clicked
    )

    if not question:
        return

    st.session_state["soc_chat_history"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    if _is_offensive(question):
        answer = OUT_OF_SCOPE_MESSAGE
        sources = []

    else:
        aggregate = _aggregate(question, df)

        if aggregate is not None:
            answer = "[CONTEXTE]\n" + aggregate
            sources = [
                "normalized_alerts/scored_findings.csv"
            ]
        else:
            retrieved = retrieve(
                question,
                finding_index,
                reports,
            )

            sources = [
                source
                for source, _ in retrieved
            ]

            answer, error = ask_llm(
                question,
                retrieved,
                st.session_state["soc_chat_history"],
            )

            if answer is None:
                answer = (
                    "[CONTEXTE]\n"
                    "La synthèse Groq est indisponible. "
                    "Voici les sources pertinentes récupérées :\n\n"
                    + "\n\n".join(
                        f"### {source}\n{text}"
                        for source, text in retrieved
                    )
                    + f"\n\nErreur : {error}"
                )

    with st.chat_message("assistant"):
        _render_answer(answer)

        if sources:
            st.caption(
                "Sources : "
                + ", ".join(sources)
            )

    st.session_state["soc_chat_history"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="SOC Assistant",
        layout="wide",
    )
    render_chatbot_tab()
