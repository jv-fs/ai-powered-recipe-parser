# AI Powered Recipe Parser

## Como usar

Utilizar este sistema é algo bem simples, exigindo apenas a presença de um ambiente python corretamente configurado e uma instalação Ollama corretamente realizada ou uma API key de algum provedor. Sugiro usar um ambiente Conda e inserir as bibliotecas necessárias.

Para instalar o Ollama, acesse https://ollama.com/ e instale o app. O modelo recomendado é o Qwen2.5-VL-7b. Instale-o usando
    
    ollama pull qwen2.5vl:7b

Caso prefira outro provedor, troque as especificações do .env seguindo o .env.example

Crie um ambiente conda com a versão Python 3.11 e instale os requisitos disponíveis no arquivo `requirements.txt`

---

Após isso, todo o funcionamento é via linha de comando. O sistema, por padrão, busca pelo Recipe.pdf caso nenhum dado seja passado para ele como argumento.

O uso típico do sistema seguirá esta linha de comando:

    python main.py --servings 6 --base-servings 1 --vegan

ou

    python main.py --scale 3 --base-servings 2 --vegan

Omitir o parâmetro `--vegan` faz com que apenas o número de porções seja modificado. Omitir parâmetros de porção e o `--vegan` faz com que o software apenas converta a entrada para .md pela rotina principal.

Caso o arquivo a ser lido não seja o `Recipe.pdf`, passar o path como argumento (primeiro parâmetro logo após o main.py), como no exemplo:

    python main.py MyCookbook.jpg --servings 6 --base-servings 1 --vegan

O output esperado será criado na pasta `results/`, contendo `output.json`, `warnings.txt` e `recipe.md`. O primeiro arquivo consiste na estrutura do que foi coletado, de modo a permitir o uso em outro local além do sistema. O segundo, é a lista de momentos em que foram registradas possíveis falhas por parte do LLM. E o último arquivo consiste na saída formatada step-by-step da receita extraída.

## Arquitetura

O programa está distribuído em 4 vertentes: ingest, extract, manipulate e output.

Ingest funciona sob as seguintes premissas:
- PDFs, .txt e imagens são os únicos inputs aceitos
- Dados bloqueados ou quebrados são recusados e warnings são gerados
- Caso estejamos falando de um PDF, será verificado seu plain text (se houver) e ele será usado.
- Caso o PDF não possua texto (ou tenha texto parcial), suas páginas são convertidas em imagens pelo PyMuPDF
- As imagens são enviadas e/ou texto são enviados para a etapa de Extraction.

Extraction funciona sob as seguintes premissas:
- Para imagens, é necessário extrair o texto delas por meio da LLM
- Então é solicitado o texto completo existente
- Após isso, é feito um recorte de cada imagem em duas metades: uma com 2/3 da imagem de cima pra baixo e outra com 2/3 da imagem de baixo para cima.
- Essas duas imagens são utilizadas para extrair o texto completo novamente e tudo é concatenado
- Compara-se as duas extrações (total ou em dois passos) e verifica-se por discrepâncias
- Se houver diferenças, pontua-se a posição e gera um warning (que será exibido no warnings.txt de saída)
- O texto polido é então enviado para a mesma LLM extrair o JSON com ingredients, tools e steps
- Cada ingredient e tool pertence a um step
- Checagens determinísticas garantem que tudo o que a LLM extrair para JSON está contido no texto polido pela dupla verificação inicial (de duas detecções) e remove duplicatas de steps distintos
    - Aqui mora o principal problema deste sistema: não foi feito muito polimento nessa saída, então quando o modelo se perde ao montar uma estrutura adequada no .md, normalmente é por conta desta etapa. Este ponto eu melhoraria o prompt e faria novas verificações duplicadas, garantindo um avanço progressivo de "Monta -> Verifica -> Monta -> verifica"
- JSON é retornado contendo tudo o que interessa, inclusive o que é confiável e o que não é (com isso avisado)

Manipulate funciona sob as seguintes premissas:
- Modificar número de servings é determinístico. É apenas calculado o coeficiente (quando é dado o número de servings na entrada ao invés dele) e multiplicado pelo valor.
- Ingredientes sem quantidade são considerados "à gosto", então não são modificados
- Veganizar exige conversão de exclusivamente os ingredientes de origem animal e são enviados para a LLM escolhida na .env (TEXT_MODEL) para tal
- Verificações nesta etapa incluem apenas barreiras para que o modelo não invente Ids ou modifique algo que não foi solicitado. Havendo discrepâncias, warnings são gerados.
- Produtos veganos são selecionados e as quantidades das carnes são deterministicamente multiplicadas as quantidades por 1.5 (foi algo definido arbitrariamente). Derivados, são mantidos em multiplicação 1.0 (como ovos ou leite).
- Resultados enviados para o output

Output funciona sob as seguintes premissas:
- Ele é construído deterministicamente, então não usa LLM em nenhum momento
- Apenas estrutura a saída em um Markdown.
- Foi feito de uma forma um tanto corrida, então é algo que eu faria um polimento mais severo se tivesse mais tempo.


## Considerações

Dado limitações de tempo de produção, este sistema não está 100% polido e pode apresentar falhas especialmente no que se refere ao campo Tools da receita. Após vários testes, constatou-se duas coisas (e a primeira um tanto óbvia): dados difíceis (especialmente os com caligrafia confusa) trazem resultados imprecisos algumas vezes, a qualidade do modelo é a âncora do resultado ser bom ou ruim (e assim constatou-se que o Qwen infere melhor textos em língua inglesa, apesar de ter feito corretamente a coleta de informações em alemão e português nos testes).

O fluxo de desenvolvimento deste sistema está brevemente documentado no arquivo `AI_log.md`, contendo um fluxo inicial da produção do software. Lá estão descritas algumas tomadas de decisão críticas e até mais de uma arquitetura do bloco Extraction, visto a mudança de direção em um dado momento da produção.

Este sistema foi construído utilizando o Agente OpenCode Big Pickle e algumas discussões de resultados com o Gemini Hydrogen.

Os textos da documentação deste sistema foram elaborados originalmente em Português e posteriormente traduzidos automaticamente (e revisados por olho humano).


