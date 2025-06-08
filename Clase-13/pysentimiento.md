Pysentimiento

https://arxiv.org/pdf/2106.09462

pysentimiento - Un Toolkit Multilingüe para Minería de Opiniones y PNL Social
pysentimiento es una librería de Python de código abierto diseñada para la minería de opiniones y tareas de Procesamiento del Lenguaje Natural de redes sociales. El objetivo principal de pysentimiento es hacer que las herramientas de vanguardia para estas tareas sean accesibles y fáciles de usar para investigadores y usuarios no expertos, especialmente en idiomas distintos al inglés.

1. Introducción y Motivación

La extracción de opiniones y estados de ánimo del texto generado por el usuario, especialmente en redes sociales, ha ganado un interés significativo. Sin embargo, los investigadores sociales enfrentan desafíos al adoptar herramientas de última generación, ya que a menudo son APIs comerciales, solo están disponibles en inglés, o son demasiado complejas para usuarios no expertos.

Problema Central: "los recursos para realizar estas tareas son escasos. Principalmente, hay que recurrir a APIs de pago proporcionadas por empresas o depender de modelos muy desactualizados o incluso no disponibles para un idioma dado que no sea el inglés."

Volumen de Datos: La inmensa cantidad de contenido generado en redes sociales ("500 millones de tweets por día son generados a nivel mundial - a 2021") hace que el análisis manual sea imposible, impulsando la necesidad de automatización.
Aplicaciones: La minería de opiniones tiene amplias aplicaciones, incluyendo el estudio del comportamiento del consumidor, campañas políticas, y la evolución de patrones emocionales (e.g., durante la pandemia de COVID-19).

2. ¿Qué es pysentimiento?

pysentimiento es un "toolkit multilingüe de Python para tareas de NLP en el ambito social". Se presenta como una solución de código abierto para las limitaciones existentes, ofreciendo modelos de vanguardia para español, inglés, italiano y portugués en una librería fácil de usar.

Fundamento Tecnológico: Está "construido sobre la librería Transformers de HuggingFace", lo que le permite aprovechar modelos de lenguaje preentrenados de última generación.
Accesibilidad: "proporciona un toolkit multilingüe fácil de usar para la minería de opiniones en redes sociales."
Disponibilidad: Está disponibilizado como "software libre y de código abierto" en GitHub y HuggingFace Hub.

3. Contribuciones Principales

El desarrollo de pysentimiento se basa en las siguientes contribuciones:

- Lanzamiento de un toolkit de código abierto y multilingüe para minería de opiniones y tareas de PNL social en Python (español, inglés, portugués e italiano).
- Provisión de una evaluación exhaustiva del rendimiento de varios modelos preentrenados de última generación para diferentes tareas de minería de opiniones en los idiomas mencionados.
- Inclusión de una pequeña evaluación de la equidad para el análisis de emociones en inglés.
- Comparación de rendimiento de pysentimiento con otras herramientas de código abierto para minería de opiniones.
Liberación de los modelos con mejor rendimiento como parte de la librería.


4. Tareas de Minería de Opiniones Soportadas

pysentimiento aborda cuatro tareas principales de minería de opiniones, con descripciones claras de cada una:

- Análisis de Sentimiento: "predecir si un texto tiene un sentimiento general —ya sea positivo, negativo o neutro." Es una de las tareas más antiguas y populares en la minería de opiniones.
- Detección de Emociones: Una tarea más compleja que el análisis de sentimiento, que intenta identificar "un estado que involucra componentes fisiológicos, subjetivos y expresivos". Se basa a menudo en las seis emociones básicas de Ekman (ira, asco, miedo, alegría, tristeza, sorpresa) o conjuntos de emociones más granulares. Requiere una "comprensión más profunda del texto" y es más subjetiva.
- Detección de Discurso de Odio: Identifica "discurso que contiene violencia hacia un individuo o un grupo de individuos, de acuerdo con ciertas características protegidas por tratados internacionales, como género, raza, idioma y otros". Su relevancia ha crecido debido a la prevalencia de estos discursos en redes sociales y sus asociaciones con el estrés y la depresión de las víctimas. Puede ser una clasificación binaria o multietiqueta.
- Detección de Ironía: Determina si un texto es irónico, donde "el significado intencionado de una declaración es lo opuesto a su significado literal". Es una tarea semántica desafiante y subjetiva, influenciada por los antecedentes culturales.
- Además, pysentimiento también considera otras tareas de NLP social como el POS tagging, el reconocimiento de entidades nombradas (NER), la detección contextualizada de discursos de odio y el análisis de sentimiento dirigido (especializado en español rioplatense), aunque no son el enfoque principal de este trabajo.

5. Metodología y Modelos Preentrenados
El proceso para seleccionar los modelos óptimos para pysentimiento implicó una evaluación exhaustiva de modelos de dominio general y especializados en redes sociales.

Preprocesamiento de Datos: Fundamental para datos de Twitter, donde abundan textos de caracter "ruidosos" y que contienen "elementos de texto no canónicos" (user names, hashtags, emojis, errores ortográficos). La estrategia adoptada incluye:
- Limitación de repeticiones de caracteres a un máximo de tres.
Conversión de handles de usuario a tokens especiales (e.g., @usuario, @USER).
- Reemplazo de hashtags por un token especial hashtag seguido del texto del hashtag, dividido en palabras si es posible.
Reemplazo de emojis por su representación textual, rodeados de un token especial emoji.

- Modelos Preentrenados: Se evaluaron diversos modelos basados en Transformers (BERT, RoBERTa, ELECTRA) y modelos especializados en redes sociales (BERTweet, RoBERTuito, AlBERTo, BERTimbau, BERTweetBR, BERTabaporu) para cada idioma. Los modelos seleccionados son "encoder-only" por su idoneidad para tareas de clasificación.

- Proceso de Fine-tuning: Los clasificadores se entrenaron con el optimizador Adam y un horario de tasa de aprendizaje triangular. Se realizó una "búsqueda exhaustiva de hiperparámetros" para cada modelo, tarea e idioma. Se realizaron diez experimentos con diferentes semillas aleatorias para una "estimación más robusta del rendimiento", reportando el Macro F1.

6. Evaluación del Rendimiento
Los resultados de la evaluación muestran que los modelos especializados en redes sociales generalmente ofrecen un rendimiento superior:

- Modelos Especializados vs. Generales: "En general, se puede ver que los modelos de lenguaje especializados para redes sociales muestran un rendimiento superior en la mayoría de los idiomas: BERTweet para inglés, BERTweetbr y BERTabaporu para portugués, y RoBERTuito para español." La única excepción fue el italiano, donde BERTit tuvo el mejor rendimiento.
- Robustez de RoBERTuito: "RoBERTuito ofrece un rendimiento robusto para la mayoría de las tareas e idiomas". Esto se atribuye a sus datos de preentrenamiento, que incluyen una cantidad sustancial de tweets en español, inglés, portugués y otros idiomas relacionados. Este modelo "podría ser un punto de partida para otros idiomas, como el catalán, el gallego o el euskera".
- Selección de Modelos: Para cada tarea e idioma, se seleccionó el modelo con mejor rendimiento. En casos de diferencias insignificantes, se prefirió el modelo monolingüe o especializado.

7. Evaluación de la Equidad
- El artículo aborda la importancia de evaluar los sesgos en los modelos de IA para evitar la amplificación de sesgos sistemáticos contra subgrupos específicos.

- Recursos Escasos: Se destaca la dificultad de encontrar recursos adecuados para análisis de equidad, que requieren información demográfica de las personas mencionadas en el texto.
- Corpus Utilizado: Se limitó el análisis a la tarea de detección de emociones en inglés utilizando el Equity Evaluation Corpus (ECC) (Kiritchenko y Mohammad, 2018). Este corpus, aunque "creado artificialmente" y pequeño, "constituye una buena guía para que cada investigador o practicante realice su propia evaluación."
Criterio de Equidad: Se usó el criterio de paridad estadística (Statistical Parity), cuantificado con la métrica de Impacto Dispar (Disparate Impact, DI). Un modelo se considera justo si DI = 1.
- Resultados de Equidad: "Para todas las combinaciones de modelo y emoción, los DIs son superiores a 0.8, lo que no proporciona evidencia de impacto adverso." Los modelos entrenados con contenido generado por el usuario (BERTweet y RoBERTuito) mostraron un comportamiento similar a los modelos de dominio general, sugiriendo que "no muestran un sesgo mayor que los modelos de dominio general."
Precauciones: Estos hallazgos deben tomarse con cautela, ya que el corpus ECC "podría no representar el contexto de despliegue real", y los modelos de lenguaje grandes han demostrado ser "altamente sesgados".

8. Comparación con Otras Herramientas
pysentimiento se comparó con otras herramientas de código abierto como VADER, TextBlob, Stanza, TweetNLP y Flair en tareas de análisis de sentimiento y detección de discurso de odio.

- Rendimiento Superior: "pysentimiento supera a las otras herramientas en la mayoría de los conjuntos de datos cuando esta comparación es posible."
- Casos Específicos: En análisis de sentimiento, TweetNLP fue el segundo mejor, superando a pysentimiento solo en el dataset Sentiment140. Flair superó a pysentimiento significativamente en el dataset SST-2, posiblemente debido a un entrenamiento específico en ese conjunto de datos.
- En detección de discurso de odio, pysentimiento superó a TweetNLP en todos los idiomas, excepto en inglés, donde TweetNLP tuvo un rendimiento considerablemente superior. Esto fue "inesperado" dado que ambas librerías usan HatEval para entrenamiento y evaluación.
- "Maldición de la Multilingüidad": Los resultados sugieren que el rendimiento subóptimo de TweetNLP en idiomas distintos al inglés "podría atribuirse a la maldición de la multilingüidad". En contraste, pysentimiento, al "seleccionar el modelo preentrenado adecuado para la tarea e idioma", logra mejores resultados que un modelo multilingüe general.

9. Conclusiones y Trabajo Futuro

- pysentimiento es un "toolkit multilingüe para extraer opiniones del texto de redes sociales" que ofrece un "rendimiento de última generación" en la mayoría de las tareas e idiomas considerados.

- Utilidad para Investigadores: Facilita el procesamiento y análisis de texto de redes sociales para investigadores.

- Fairness como Procedimiento: A pesar de las limitaciones del análisis de equidad, proporciona un "procedimiento paso a paso de cómo los profesionales pueden diagnosticar sesgos relevantes para su contexto de aplicación antes del despliegue, previniendo así posibles daños a la población objetivo de esta herramienta."
- Disponibilidad: El código y los modelos están disponibles públicamente en GitHub y HuggingFace Hub.
Planes Futuros: Se planea "extender pysentimiento a otros idiomas y tareas, y también proporcionar más utilidades de extracción de información", así como "analizar información contextualizada y no solo oraciones aisladas".

- En resumen, pysentimiento representa un avance significativo en la democratización de las herramientas de minería de opiniones multilingües, con un fuerte enfoque en el rendimiento y una consideración inicial sobre la equidad de los modelos.

