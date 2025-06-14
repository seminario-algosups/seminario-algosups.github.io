# Trabajo Práctico #3

En grupos de hasta 3 personas, deberán tomar la [noticia sugerida](../Clase-13/noticia1.txt) de la clase 13  y anotar las entidades `PERSON, GPE, DATE` en el cuerpo del texto, envolviendo cada una de las entidades de la siguiente forma: `[valor de la entidad](TIPO DE ENTIDAD)`. 

Utilizar la siguiente definición de entidades: 
    
*   Usar `PERSON` para personas.
*   Usar `GPE` para lugares (ciudades, países).
*   Usar `DATE` para fechas.
*   Las entidades NO deben incluir puntuación al inicio ni al final (paréntesis, comas, puntos, etc.)

Además, deberán entregar un archivo de criterios de anotación donde justifiquen las decisiones tomadas y, si las hubiera, también las excepciones. 

Pueden usar como referencia la [notebook](../Clase-13/entities-2.ipynb) que veremos en la clase 13. 

En esta notebook van a encontrar el código necseario para usar sus anotaciones como entrenamiento para un NER custom.

**NO ES NECESARIO EJECUTAR ESTE CÓDIGO NI ENTRENAR EL NER  como parte de la entrega.** 


{% include copybutton.html %}

{% include additional_content.html %}
