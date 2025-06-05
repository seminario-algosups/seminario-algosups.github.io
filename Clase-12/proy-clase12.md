
**Materiales:**

*   Presentación (Diapositivas).
*   Entorno Python con Jupyter Notebooks o similar.
*   Librerías: `scikit-learn`, `gensim`, `spacy`, `stanza`.
    *   `pip install scikit-learn gensim spacy stanza`
    *   `python -m spacy download es_core_news_sm` (o el modelo en inglés que prefieras)
    *   En Python, después de `import stanza`: `stanza.download('es')` (o 'en')

---

**Estructura de la Clase:**

**Parte 1: Introducción al NLP y la Importancia de la Semántica (15 minutos)**

*   ¿Qué es el Procesamiento del Lenguaje Natural (NLP)?
*   La diferencia entre sintaxis y semántica.
*   ¿Por qué es importante medir la similitud semántica y extraer información estructurada (entidades, relaciones)?
    *   Aplicaciones: Motores de búsqueda, sistemas de recomendación, chatbots, análisis de sentimiento, resumen automático, construcción de grafos de conocimiento.

**Parte 2: Métodos de Medición de Similitud Semántica (75 minutos)**

*   **Concepto Clave:** Representación vectorial de texto (Vector Space Models).
    *   La idea de convertir texto en números para poder aplicar operaciones matemáticas.

*   **ii.1) Similitud Coseno:**
    *   **Explicación:** Mide el coseno del ángulo entre dos vectores. Valores entre -1 y 1 (o 0 y 1 para vectores no negativos). Más cercano a 1 = más similar.
    *   **Representaciones Vectoriales Comunes (previo al coseno):**
        *   **Bag of Words (BoW):** Frecuencia de palabras.
        *   **TF-IDF (Term Frequency-Inverse Document Frequency):** Pondera la importancia de las palabras.
    *   **Fórmula (conceptual):** `cos(θ) = (A · B) / (||A|| ||B||)`
    *   **Ventajas:** Simple, efectivo con representaciones como TF-IDF para similitud léxica.
    *   **Desventajas:** Con BoW/TF-IDF, no captura el significado si las palabras son diferentes (sinónimos), ni el orden de las palabras.
    *   **Demostración Práctica (Python con `sklearn`):**
        ```python
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        doc1 = "el rápido zorro marrón salta sobre el perro perezoso"
        doc2 = "un zorro veloz brinca encima del can cansado"
        doc3 = "el gato juega con la pelota"

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([doc1, doc2, doc3])

        # Similitud entre doc1 y doc2
        sim_1_2 = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        print(f"Similitud Coseno (TF-IDF) entre doc1 y doc2: {sim_1_2[0][0]:.2f}")

        # Similitud entre doc1 y doc3
        sim_1_3 = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[2:3])
        print(f"Similitud Coseno (TF-IDF) entre doc1 y doc3: {sim_1_3[0][0]:.2f}")
        ```

*   **ii.2) Similitud por Presencia de Token (Ej. Jaccard Index):**
    *   **Explicación:** Se basa en la superposición de tokens (palabras) entre dos textos.
    *   **Jaccard Index:** `|Set1 ∩ Set2| / |Set1 ∪ Set2|`. (Tamaño de la intersección / Tamaño de la unión).
    *   **Ventajas:** Muy simple de calcular e interpretar. Útil para detectar duplicados exactos o muy cercanos.
    *   **Desventajas:** Ignora completamente la semántica (sinónimos) y la frecuencia de los tokens. "Coche" y "Automóvil" tendrían 0 similitud.
    *   **Demostración Práctica (Python):**
        ```python
        def jaccard_similarity(text1, text2):
            set1 = set(text1.lower().split())
            set2 = set(text2.lower().split())
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union != 0 else 0

        print(f"Similitud Jaccard entre doc1 y doc2: {jaccard_similarity(doc1, doc2):.2f}")
        print(f"Similitud Jaccard entre doc1 y doc3: {jaccard_similarity(doc1, doc3):.2f}")
        ```

*   **Introducción a Word Embeddings (Contexto para Word2Vec y FastText):**
    *   La idea de que "una palabra se conoce por la compañía que mantiene" (Firth).
    *   Representaciones densas (vectores con muchos números no cero) que capturan el contexto y significado.
    *   Permiten realizar operaciones semánticas: `rey - hombre + mujer ≈ reina`.

*   **ii.3) Word2Vec (Google):**
    *   **Explicación:** Modelo predictivo para aprender embeddings de palabras a partir de un corpus grande.
    *   **Arquitecturas Principales:**
        *   **CBOW (Continuous Bag of Words):** Predice la palabra actual basándose en su contexto.
        *   **Skip-gram:** Predice las palabras del contexto basándose en la palabra actual.
    *   **Similitud:** Una vez entrenado, cada palabra tiene un vector. La similitud entre palabras (o documentos promediando/sumando vectores de palabras) se calcula típicamente con similitud coseno.
    *   **Ventajas:** Captura relaciones semánticas y sintácticas. Modelos pre-entrenados disponibles.
    *   **Desventajas:** No maneja bien palabras fuera de vocabulario (OOV) si no están en el entrenamiento.
    *   **Demostración Práctica (Python con `gensim` - usando modelos pre-entrenados o uno pequeño de ejemplo):**
        *   Mostrar cómo cargar un modelo y obtener vectores/similitud. (Puede ser complejo para una demo corta si no se tienen modelos listos).
        *   Conceptualmente: "Calculamos el vector promedio de las palabras en cada documento y luego aplicamos similitud coseno".

*   **ii.4) FastText (Facebook):**
    *   **Explicación:** Extensión de Word2Vec. Representa cada palabra como una bolsa de n-gramas de caracteres.
    *   **Ejemplo:** La palabra "manzana" con n=3 sería `<ma, man, anz, nza, zan, ana, na>`. El vector de "manzana" es la suma de los vectores de estos n-gramas.
    *   **Ventajas:**
        *   Maneja palabras OOV construyendo su vector a partir de sus n-gramas.
        *   Suele funcionar mejor para lenguajes morfológicamente ricos y para palabras raras.
    *   **Similitud:** Similar a Word2Vec, se usa similitud coseno sobre los vectores de palabras/documentos.
    *   **Demostración Práctica (Python con `gensim` o `fasttext` library):**
        *   Similar a Word2Vec, mostrar carga y uso.
        *   *Nota:* `spaCy` usa embeddings similares a FastText (subword features) en sus modelos grandes.

*   **Comparativa y Cuándo Usar Qué:**
    *   **Presencia de Token:** Rápido, para duplicados o similitud léxica muy alta.
    *   **TF-IDF + Coseno:** Bueno para recuperación de información, similitud basada en palabras clave importantes.
    *   **Word Embeddings (Word2Vec, FastText) + Coseno:** Para similitud semántica profunda, comprensión de sinónimos, analogías. FastText mejor con OOV y morfología.

**Parte 3: Reconocimiento de Entidades Nombradas (NER) y Extracción de Relaciones (75 minutos)**

*   **5.b.i) Sistemas de Anotación:**
    *   **¿Qué es NER?** Identificar y clasificar entidades nombradas en el texto (PERSON, ORG, LOC, DATE, GPE, etc.).
    *   **¿Por qué anotar?** Para entrenar modelos de NER supervisados, necesitamos datos etiquetados.
    *   **Esquema IOB (o IOB2):**
        *   **I - Inside:** El token está dentro de una entidad.
        *   **O - Outside:** El token no es parte de ninguna entidad.
        *   **B - Beginning:** El token es el inicio de una entidad (usado cuando dos entidades del mismo tipo son adyacentes).
        *   **Ejemplo:**
            ```
            Alex    B-PER
            Smith   I-PER
            trabaja O
            en      O
            Google  B-ORG
            en      O
            Londres B-LOC
            .       O
            ```
    *   **Esquema BILOU (o BIOES):**
        *   **B - Beginning:** Inicio de una entidad multi-token.
        *   **I - Inside:** Dentro de una entidad multi-token.
        *   **L - Last:** Fin de una entidad multi-token.
        *   **O - Outside:** No es parte de una entidad.
        *   **U - Unit:** Entidad de un solo token.
        *   **Ejemplo:**
            ```
            Alex    B-PER
            Smith   L-PER
            trabaja O
            en      O
            Google  U-ORG
            en      O
            Londres U-LOC
            .       O

            Nueva   B-LOC
            York    L-LOC
            es      O
            grande  O
            ```
    *   **Ventajas de BILOU:** Más expresivo, puede ayudar a los modelos a aprender mejor los límites de las entidades.

*   **5.b.ii) NER de SpaCy y Stanza:**

    *   **SpaCy:**
        *   **Descripción:** Librería de NLP "industrial", rápida, eficiente, con modelos pre-entrenados en varios idiomas.
        *   **Uso Básico:**
            ```python
            import spacy

            # Cargar modelo (ej: español pequeño)
            # python -m spacy download es_core_news_sm
            nlp_spacy = spacy.load("es_core_news_sm") # o "en_core_web_sm"

            texto = "Apple está buscando comprar una startup del Reino Unido por mil millones de dólares. Tim Cook visitará Londres la próxima semana."
            doc_spacy = nlp_spacy(texto)

            print("Entidades con spaCy:")
            for ent in doc_spacy.ents:
                print(f"- Texto: {ent.text}, Etiqueta: {ent.label_}, Inicio: {ent.start_char}, Fin: {ent.end_char}")
            ```
        *   **Características:** Acceso a `ent.text`, `ent.label_`, `ent.start_char`, `ent.end_char`.
        *   **Visualización:** `spacy.displacy.render(doc_spacy, style="ent", jupyter=True)`

    *   **Stanza (StanfordNLP):**
        *   **Descripción:** Librería de NLP desarrollada por el grupo de Stanford, conocida por su alta precisión y soporte multilingüe, a menudo basada en redes neuronales profundas.
        *   **Uso Básico:**
            ```python
            import stanza

            # Descargar modelo para español (solo la primera vez)
            # stanza.download('es') # o 'en'
            nlp_stanza = stanza.Pipeline('es', processors='tokenize,ner') # o 'en'

            doc_stanza = nlp_stanza(texto)

            print("\nEntidades con Stanza:")
            for ent in doc_stanza.ents:
                print(f"- Texto: {ent.text}, Tipo: {ent.type}")
            ```
        *   **Características:** Acceso a `ent.text`, `ent.type`.
        *   **Comparación:** Stanza puede ser más preciso en algunos casos pero más lento que spaCy. La elección depende de los requisitos del proyecto (velocidad vs. precisión, idiomas).

*   **5.b.ii) Extracción de Relaciones (Introducción):**
    *   **¿Qué es?** Identificar relaciones semánticas entre las entidades nombradas detectadas.
        *   Ej: `(Tim Cook, CEO_de, Apple)`, `(Apple, localizada_en, Reino Unido)`
    *   **Importancia:** Construir grafos de conocimiento, responder preguntas complejas, análisis profundo de texto.
    *   **¿Cómo se relaciona con NER?** NER es un paso previo fundamental. Primero identificas las "cosas" (entidades), luego cómo se relacionan.
    *   **Enfoques Comunes (mencionar brevemente):**
        *   **Basados en Reglas/Patrones:** Definir patrones léxico-sintácticos (ej. `ENTIDAD1 <verbo_relacion> ENTIDAD2`).
        *   **Supervisados:** Entrenar clasificadores con datos anotados de relaciones.
        *   **Distant Supervision:** Alinear texto con bases de conocimiento existentes para generar datos de entrenamiento automáticamente (puede ser ruidoso).
        *   **Open Information Extraction (OpenIE):** Extraer tuplas (sujeto, predicado, objeto) sin un esquema predefinido.
    *   **Ejemplo Conceptual (no código complejo, sino idea):**
        *   Texto: "Tim Cook, CEO de Apple, visitó Cupertino."
        *   NER: Tim Cook (PER), Apple (ORG), Cupertino (LOC).
        *   Posible Relación: `(Tim Cook, es_CEO_de, Apple)`, `(Tim Cook, visitó, Cupertino)`.
    *   **Herramientas:**
        *   SpaCy tiene componentes experimentales o extensiones comunitarias para RE.
        *   Stanza también tiene capacidad para análisis de dependencias que son cruciales para muchos enfoques de RE.
        *   Otras librerías especializadas: OpenNRE, etc.

**Parte 4: Conclusiones y Próximos Pasos (15 minutos)**

*   Resumen de los conceptos clave aprendidos.
*   Importancia de elegir el método/herramienta adecuado según la tarea.
*   Retos en NLP: Ambigüedad, contexto, escasez de datos etiquetados.
*   Áreas de estudio futuras: Sentence Embeddings (BERT, Sentence-BERT), Modelos de Lenguaje Grandes (LLMs), construcción de Grafos de Conocimiento.
*   Sesión de Preguntas y Respuestas.

---

**Consejos para la Clase:**

1.  **Demostraciones en Vivo:** Son cruciales. Prepara los notebooks con antelación.
2.  **Ejemplos Claros:** Usa frases sencillas y variadas para ilustrar los conceptos.
3.  **Interactividad:** Fomenta preguntas. Puedes plantear pequeños "retos" a la audiencia.
4.  **Foco Práctico:** Aunque la teoría es importante, muestra cómo se aplican estos conceptos con código.
5.  **Recursos Adicionales:** Proporciona enlaces a documentación, tutoriales, artículos.
6.  **Simplifica:** Algunos conceptos (como el entrenamiento de Word2Vec) pueden ser muy densos. Enfócate en el "qué hacen" y "cómo se usan" los modelos pre-entrenados para una clase introductoria.

---

Espero que esta estructura te sea de gran utilidad. ¡Mucho éxito con tu clase!