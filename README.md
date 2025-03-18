{% include head.html %}

## Presentación

Este espacio está pensado para servir de soporte al seminario "Algoritmos supervisados y convenciones de anotación para tareas de procesamiento del lenguaje natural", dictado durante el primer cuatrimestre de 2025 para la carrera de [Letras](https://letras.filo.uba.ar/) de la [Facultad de Filosofía y Letras](https://www.filo.uba.ar/) de la [Universidad de Buenos Aires](https://uba.ar/). El seminario está a cargo de [Fernando Carranza](https://fernando-carranza.github.io/) como profesor adjunto interino con dedicación simple y [Fernando Schiaffino](https://ar.linkedin.com/in/fernando-schiaffino-339237a1/es) con asignación de funciones.

## Modalidad

En la cursada vamos a utilizar el [campus](https://campus.filo.uba.ar/) para disponibilizar la bibliografía y para comunicarnos a través del foro. El resto de los materiales de clase se van a centralizar en esta página y su respectivo repositorio. El repositorio se puede clonar para correr el código de las clases de manera local, siempre y cuando se haya instalado correctamente antes Jupyter Notebook, Python y las librerías relevantes. También es posible correr el código desde una cuenta de Google Colab.

## Cronograma de clases y materiales

<table>
  <tr>
    <th>Clase</th>
    <th>Fecha</th>
    <th>Docente</th>
    <th>Temas</th>
    <th>Materiales</th>
  </tr>
  <tr>
    <td><a href="./Clase-01/index.md">01</a></td>
    <td>22/03/2025</td>
    <td>Carranza</td>
    <td>
        <ul>
            <li>Presentación seminario</li>
            <li>1.i) la inteligencia artificial</li>
            <li>1.ii) programación clásica vs. el aprendizaje automático (*machine learning*).</li>
            <li>1.iii) Procesamiento del lenguaje natural: definición, tareas y enfoques;</li> 
            <li>1.iv) lenguaje de programación Python en Google Colab (tipos de datos, variables, sintaxis básica)</li>
        </ul>
    </td>
    <td>
        <ul>
            <li></li>
        </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-02/index.md">02</a></td>
    <td>29/03/2025</td>
    <td>Schiaffino</td>
    <td>
        <ul>
            <li>1.iv) lenguaje de programación Python en Google Colab (funciones, librerías, archivos)</li>
            <li>1.v) Exploración de algunos recursos de libre acceso relevantes: Kaggle y Huggingface, GitHub.</li>
        </ul>
    </td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-03/index.md">03</a></td>
    <td>05/04/2025</td>
    <td>Schiaffino</td>
    <td>
        <ul>
    	<li>2.i) Aprendizaje supervisado y no supervisado.</li>
    	<li>2.ii) Métricas usuales para medir el rendimiento de modelos de clasificación (accuracy, precisión y cobertura).</li>
    	<li>2.iii) Anotación como tarea a resolver por un modelo predictivo.</li>
    	<li>2.iv) Datos estructurados y no estructurados: manejo de estructuras de almacenamiento de datos (json, csv)</li>. 
        </ul>
    </td>
    <td>
    <ul>
    </ul>
            <li></li>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-04/index.md">04</a></td>
    <td>12/04/2025</td>
    <td>Schiaffino</td>
    <td>
      <ul>
        <li>2.v) Vectorización (conversión de un texto en tanto dato no estructurado en un arreglo numérico estructurado; CountVectorizer, TfidfVectorizer).</li>
<li>2.vi) Modelos de clasificación: Bayesiano ingenuo.</li>
      </ul>
    </td>
    <td>
      <ul>
      	<li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-05/index.md">05</a></td>
    <td>19/04/2025  <br> (Semana Santa) <br>clase virtual</td>
    <td>Schiaffino</td>
    <td>
      <ul>
        <li>2.vi) Modelos de clasificación: Regresión Logística, Máquina de soporte vectorial (<i>Support Vector Machines</i>).</li>
      </ul>
    </td>
    <td>
      <ul>
    <li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-06/index.md">06</a></td>
    <td>26/04/2025</td>
    <td>Carranza</td>
    <td>
      <ul>
        <li>3.a.i) nociones básicas de morfología: raíz, lema, tema y forma de palabra, morfología flexiva y derivativa;</li>
<li>3.a.iv) etiquetamiento en CONLL-U: columnas de ID, de palabra y de lema.</li> 
      </ul>
    </td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-07/index.md">07</a></td>
    <td>03/05/2025 <br> (Fin de semana largo <br>por día del trabajador)<br> clase virtual</td>
    <td>Schiaffino</td>
    <td>
      <ul>
        <li>3.a.ii) tokenización, stemming y lemmatizing de NLTK;</li>
        <li>3.a.iii) lemmatizer de Spacy basado en reglas y basado en aprendizaje automático</li>
        <li>TP1 en clase</li>
      </ul>
    </td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-08/index.md">08</a></td>
    <td>10/05/2025</td>
    <td>Carranza</td>
    <td>
      <ul>
        <li>3.a.ii) Postag NLTK con tagset BNC</li>
<li>3.b.i) etiquetas de BNC</li>
<li>3.b.ii) Postag NLTK con tagset Universal y postag de Spacy y Stanza</li>
<li>3b.i) etiquetas de Universal. Práctica de anotación en clase de Universal con CONLL-U con una noticia.</li>
      </ul>
    </td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-09/index.md">09</a></td>
    <td>17/05/2025</td>
    <td>Carranza</td>
    <td>
      <ul>
        <li>3.b.ii) etiquetamiento en CONLL-U de clase de palabra con EAGLES y de rasgos morfológicos</li>
        <li>4.i) Análisis basado en constituyentes y gramáticas basadas en dependencias.</li>
      </ul>
    </td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-10/index.md">10</a></td>
    <td>24/05/2025</td>
    <td>Carranza</td>
    <td>
      <ul>
        <li>4.iii) Penn Treebank. Sistema de anotación sintáctica basada en constituyentes.</li>
<li>4.ii.a) Parser basado en constituyentes BLLIP</li>
<li>4.ii.b) parsers basados en dependencias Spacy y Stanza.</li>
<li>4.iv) Análisis sintáctico basado en dependencias y su anotación en CONLL-U.</li>
      </ul>
    </td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-11/index.md">11</a></td>
    <td>31/05/2025</td>
    <td>Carranza<br>Schiaffino</td>
    <td>
      <ul>
        <li>4.iv) Análisis sintáctico basado en dependencias y su anotación en CONLL-U.</li>
        <li>4.ii) Representación del texto como bolsa de palabras.</li>
        <li>iii) Representación del texto como arreglo numérico: matrices ralas (término-término, término-documento) y matrices densas (embeddings).</li>
        <li>Entrega consigna de TP2</li>
      </ul>
    </td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-12/index.md">12</a></td>
    <td>07/06/2025</td>
    <td>Schiaffino</td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-13/index.md">13</a></td>
    <td>14/06/2025</td>
    <td>Carranza</td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-14/index.md">14</a></td>
    <td>21/06/2025</td>
    <td>Schiaffino</td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
    <td>
      <ul>
        <li></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><a href="./Clase-15/index.md">15</a></td>
    <td>28/06/2025</td>
    <td>Carranza</td>
    <td>
      <ul>
        <li>Cierre de cursada.</li>
      </ul>
    </td>
    <td></td>
  </tr>
</table>

{% include change_href.html %}

{% include additional_content.html %}
