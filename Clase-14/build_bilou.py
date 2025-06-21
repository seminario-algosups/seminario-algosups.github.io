import spacy
import pandas as pd
from typing import Optional

def annotate_df(
    text: str,
    nlp: Optional[spacy.Language] = None,
    model: str = "es_core_news_sm"
) -> pd.DataFrame:
    """
    Tokeniza `text`, extrae entidades y POS, y devuelve un DataFrame:
    ┌────────┬────────┬────────┐
    │ token  │   pos  │  ner   │  ← BILOU
    └────────┴────────┴────────┘
    
    Params
    -------
    text  : Texto a procesar.
    nlp   : (opcional) pipeline spaCy ya cargado.
    model : Nombre del modelo a usar si no se pasa `nlp`.
    
    Returns
    -------
    pandas.DataFrame
    """
    # 1) Cargamos el pipeline sólo una vez si no lo trae quien llama
    if nlp is None:
        nlp = spacy.load(model)

    doc = nlp(text)

    # 2) Preparamos una lista BILOU con "O" por defecto
    bilou = ["O"] * len(doc)

    # 3) Recorremos cada entidad para marcar B, I, L o U
    for ent in doc.ents:
        if len(ent) == 1:
            bilou[ent.start] = f"U-{ent.label_}"
        else:
            bilou[ent.start]       = f"B-{ent.label_}"
            for i in range(ent.start + 1, ent.end - 1):
                bilou[i]           = f"I-{ent.label_}"
            bilou[ent.end - 1]     = f"L-{ent.label_}"

    # 4) Convertimos todo a filas de un DataFrame
    rows = [
        {"token": tok.text, "pos": tok.pos_, "ner": bilou[i]}
        for i, tok in enumerate(doc)
    ]
    return pd.DataFrame(rows)