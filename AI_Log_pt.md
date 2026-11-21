# AI Log

Este log está estruturado indicando o caminho adotado na confecção do código usando o OpenCode como AI Agent. Foi escrito originalmente em Português e posteriormente convertido para inglês. A escolha da linguagem original é pela maior intimidade do desenvolvedor com a sua língua nativa, permitindo uma articulação melhor na escolha de palavras.

1. Iniciei o processo selecionando as tecnologias que eu poderia usar, buscando por aquelas que fariam sentido no escopo do cookbook. Primeiro li a respeito de OCR e de como ele é propriamente utilizado nos dias de hoje. A tecnologia antiga utilizava padrões visuais para identificar letras e texto em imagens, o que não tinha uma precisão tal qual hoje se tem usando LLMs ou mesmo estruturas mais simples como o Tesseract (OCR puro).
Minha escolha foi trabalhar de uma maneira híbrida, não somente com as LLMs fortíssimas de hoje em dia. E o motivo? Bom, na primeira versão pensada para este software eu me dei por conta de que eu poderia fazer diversas verificações posteriores após uma rede muito boa, como a do Qwen2.5, extrair todo o texto de um PDF ou imagem. Contudo, se o Qwen errasse na largada, todas as próximas verificações seriam em vão (garbage in, garbage out). A solução então? Dobrar a coleta. Só que se eu colocasse duas LLMs paralelas, pensando na ideia de rodar tudo localmente, ficaria algo muito caro em termos de Hardware para manter, considerando o uso em um PC de entrada.

Considerei a ideia de plugar uma API de uma dessas redes grandes por aí, como o Gemini ou o GPT, mas fiz um paralelo com a segurança de dados além de que, para a maior parte dos casos, isso seria uma solução paga. E se a receita fosse sigilosa? E se a transcrição fosse de uma receita de um livro centenário que confere um segredo de família? Melhor fazer isso localmente.

E claro, não menos importante, a linguagem escolhida foi Python!

2. Com a ideia de extração inicial, conversei a respeito da arquitetura com o modelo Big Pickle, da OpenCode, escolhido como a melhor opção gratuita de Code Agent que eu consegui me adaptar. A ideia foi discutir a implementação para começar com o primeiro bloco do sistema: ingest.py
Durante esta conversa, me deparei em alguns pontos interessantes sobre os quais refleti a respeito:
    - Se a entrada for um PDF e este PDF for parcialmente construído por meio de imagens, é prudente converter a página inteira do PDF em imagens totais de modo que se houver alguma informação importante nas próprias imagens, em meio ao texto puro, elas não sejam perdidas.
    - Contudo, se houver texto puro no PDF, é interessante usar ele como verificação de novas extrações, isto é, se eu tenho parte do PDF escaneado e parte digitada, eu não quero perder a parte determinística (digitada) por erros de halucinação. Faz mais sentido comparar os dados do texto e os de uma detecção pelo Qwen. Se o Qwen divergir, assume-se os dados em texto. Além do fato de ser possível dar ao modelo o texto puro para referência pré-detecção.
        - Se eu tenho as imagens das páginas do PDF, é inteligente modificar o contraste pelo OpenCV, de modo a deixar palavras mais nítidas para facilitar a detecção do modelo.
        - Sobre este ponto, eu discuti com o modelo e decidi abortar essa ideia. A sugestão que me veio dele, associado ao OpenCV, foi de usar métodos de EXIF e leve CLAHE para aumento de nitidez. Contudo, comecei a pensar que este projeto inicial estivesse muito profundo e preferi manter sem o auxílio do OpenCV.
    - E claro que se eu já ganhei o input em imagens, só trato. E também, se eu já recebi o input como um .txt ou texto puro, não há tratamento necessário.

O techStack dessa fase foi então: biblioteca PyMuPDF e OpenCV

Decisões importantes de alguns detalhes:
    - Se a leitura de uma página do PDF falhar, o que acontece? Optei por descartar a página e exibir resultados parciais com um warning. Apesar disso não ser uma receita completa, ainda parece ter algum significado.
    - Quando o input é um conjunto de imagens, o que acontece? Respeito a ordem dada na entrada e leio com OpenCV.

3. Com as decisões da arquitetura inicial do ingest.py, decidi ir adiante para a construção do extract.py. Este segundo arquivo é o responsável por chamar a LLM e entregar o pacote de saída IngestResult para extração.
Iniciei matutando a ideia de usar um OCR para gerar texto de apoio na inferência da LLM. Contudo, desisti da ideia de enviar a inferência junto do conteúdo original do input para a LLM com medo de ancorar a inferência. Cogitei mascarar os dados com tags de número ou nome ou outro dado, mas considerei essa hipótese fraca demais.

Depois de uma reflexão, decidir ir pela abordagem em que a LLM é o corretor do OCR. Peguei o RapidOCR como biblioteca anexável ao projeto (escolhido pelo fácil download via pip). Cogitei, ainda, o uso do Tesseract, mas eliminei por ser necessário baixar binários externamente (a escolha foi mais por uma questão de praticidade do que necessariamente por eficiência ou qualidade do resultado). Então a lógica virou a seguinte:
    - OCR e LLM inferem sobre o texto.
    - LLM retorna um JSON com os dados estruturados corretamente.
    - Inicia-se um processo de comparação entre a saída do OCR e do LLM.
    - Se as duas saídas forem iguais, é um match, e aquele dado é aceito.
    - Se forem distintas, utiliza-se a bbox do OCR para extrair a imagem do arquivo original e envia essa informação novamente para o OCR e para o LLM isoladamente. Se o resultado for o mesmo, match. Se for diferente, é dito unresolved e gera warning.
    - Se um trecho é terminado como unresolved, decide-se o valor com melhor score (seja o do OCR, seja o do LLM), apenas para seguir com as demais etapas. Só que daqui pra frente já é considerado que o resultado é parcial, visto que falhou.

Pensei nessa ótica para servir como uma espécie de freio. Entendi que o OCR seria capaz de limitar a capacidade elevada do LLM, de modo a evitar que ele deduza tudo o que entender pouco (ou alucine no processo). A verificação dupla (critério de desempate), entretanto, é um gargalo de hardware bem grande dessa abordagem. Dentre todas as opções, associo a leitura da imagem a parte mais importante de todo o software, visto que é aqui que existe a possibilidade alta da entrada ser ruim (garbage in, garbage out). Se a entrada for ruim, todo o resto do software é essencialmente inútil.

Creio que deva existir abordagens mais otimizadas para a solução deste problema, mas dentre o que eu pesquisei e conversei com o opencode, foi o que soou mais razoável dentro do escopo do projeto. Resultados melhores exigiriam mais tempo de desenvolvimento e considerei a segurança dessa abordagem como suficiente para a coding challenge.

4. Tive problemas de truncamento na saída do modelo. Optei por reduzir mais ainda o tamanho da imagem de entrada para sobrar mais tokens para a saída. Outra abordagem seria mudar o limite do Ollama dando mais tokens e contexto, mas decidi ir pelo caminho de redução da imagem mesmo.

5. Tive mais problemas associados aos formatos retornados pelo modelo. O schema não estava encaixando e dados acabavam sendo descartados no processo. Isso fazia com que a recipe terminasse incompleta sempre. Ajustes associados ao prompt foram realizados, clarificando formatos

6. Notei que os resultados preliminares não foram bons o suficiente e decidi mudar o raciocínio da extração. Excluí as modificações preliminares e fiz um modelo inicial de múltiplos arquivos para Extract. A pasta surgiu e foi parar no Git logo em seguida.

7. Refiz a engenharia básica da arquitetura, frisando exatamente o que foi descrito no item 3 deste log, tentando resultados melhores. Percebi que algumas colocações minhas no chat do modelo foram vagas e ele inferiu coisas na arquitetura que ultrapassavam a especificação que eu desenvolvi. Por isso o descarte do código original foi a decisão correta. Refiz com mais calma e resolvi colocar toda a especificação no SPEC.md e zerar o chat dessa etapa de implementação. OpenCode leu o SPEC.md reestruturado e aplicou as correções adicionais na nova estrutura de arquivos.... E deu errado.

8. Decidi mudar tudo novamente, começando do zero e excluindo a ideia de usar OCR. Comparei as duas extrações e essa foi a diferença:

LLM:

189 Will man den Braten nach dem 1. Tag oder 2. Tag gebrauchen, so giept man das Essig- wasser heiß darüber. Soll der Braten erst nach 8 Tagen zubereitet werden, so giept man es kalt darauf. Zum Anbraten den Speck auslassen, Butter dazugeben, dann gibt man den abgetrockne- ten Braten ins heiße Fett. Wenn er von al- len Seiten knusprig angebraten ist, giept man etwas Essigwasser hinzu. Auch kann man einige Äpfel, Möhren und Wacholderbeeren in den Topf geben. Man läßt den Braten langsam mit den Zutaten gar- brötscheln. Für die Soße schwitzt man den Bratenfond mit Mehl an, füllt mit Essigwasser und Rot- wein auf, gibt dann gequollene Korinthen u. Rosinen hinzu und soviel Zuckcr, daß die Soße einen säuerlich-süßen Geschmack bekommt. Man legt den Braten, wenn er noch nicht ge- schnitten ist, eine Stunde vor dem Anrich- ten in die Soße, damit er den Geschmack derselben bekommt. Rotwein Korinthen

OCR:

189 Willman den Braten nach h dem 1.Tag oder es'kall darauy. auslassen,BuHer tenBraten ins heiBe fet. wenn er von al- len Serlen Rnusprig qn gebraten ist, giepk man elwas Essigwa sser yinzu brotscheln. fir dieSo/3e Schwitet nan den Bratentond mi Hehl qn tilltnit tssigwasserundRot weinquy qibt dann gequollene Korinthen u. Rosinen' hinee und Soviel. Zudker, da/3die Soybeeinen - sifBen Geschmack bekommt. ManlegtdenBraten wenn er noch hnichtge SchniHenist eine Stunde vor dem Anrid ten in die Sopbe , damib er den geschmack derselbenbekommt. ROEwg Kori then

A qualidade obtida na elaboração com LLM pura foi muito melhor do que a do OCR... Isso significa que o OCR não freia a LLM... Ele simplesmentes destrói a qualidade da inferência da LLM. Vou começar novamente sem OCR e com outro método de detecção.

9. O novo método foi o seguinte:
    - Agora usamos apenas a LLM para toda a inferência, mas com duas camadas de detecção.
    - A primeira camada é uma passada geral extraindo o texto de todo o documento
    - A segunda camada consiste em quebrar cada imagem em porções de 2/3: um crop da imagem contendo 2/3 dela de cima pra baixo e outro crop da imagem contendo 2/3 dela de baixo pra cima.
    - Deterministicamente, por Python puro, compara-se os dois dados. Se houve pequenas discrepâncias entre alguns dos dados, é emitido um warning para cada uma das discrepâncias. Aceita-se dados iguais ou simplesmente assume que a coleta das imagens cortadas é a resposta correta.
    - Envia-se um prompt com o texto final para a mesma LLM solicitando a conversão para JSON. A engenharia aqui é meramente prompt e foi a solução mais simples que eu consegui pensar.
    - Após retornar o JSON, verifica-se duplicatas que estejam no mesmo step e se a quote está contida no texto de entrada antes do JSON (garantindo que não ocorreram alucinações)
    - Retorna este JSON para a etapa de Manipulação.

    Os primeiros testes foram bem promissores e realmente houve distinções significativas entre os resultados cropados e os resultados de página inteira. Assumi esta versão como a ideal.
    Ponto negativo foi o tempo de inferência. Senti que por rodar várias vezes a LLM em busca de resultados, o tempo de processamento aumentou muito. Como meu hardware é limitado, foi um ônus desta implementação.

10. Depois do estresse e tempo de aproximadamente 4 horas de desenvolvimento da extração (dada as diversas mudanças de arquitetura adotadas, nem todas explicitamente referenciadas neste log), parti para a fase de Manipulação. Acreditei que as próximas duas etapas me demandariam bem menos tempo do que a etapa de extração, uma vez que verificações mais determinísticas fariam sentido. Aproveitei o fluxo e já implementei junto o output.

Até o momento eu vinha utilizando "main" fictícia para testar o código além dos testes de unidade automatizados e inferências da própria OpenCode a partir de questionamentos que eu realizei. A implementação alocou uma main.py também para facilitar testes de sistema e identificação de bugs gerais. Fiz isso tudo meio junto para fechar o pacote certinho e só aperfeiçoar trechos isolados.

O output foi bem determinístico (tanto que não usei processamento de LLM nele). O output é em formato de listas dentro de um .md. A escolha da formatação da .md, visto que seria algo puramente visual e determinístico, eu deixei nas mãos do meu agente. Preservei as saídas output.json e warnings.txt para ter a rastreabilidade de erros e permitir qualquer outro destino dos dados coletados nesse processo todo.

11. Explicando melhor o que eu fiz na parte de manipulação, adotei essa sequência:
    - Converto os dados numéricos que eu recebi para uma forma puramente numérica (sem caracteres unicode de 1/2 ou 1/8 ou algo do tipo)
    - Deterministicamente defino se a escalabilidade da receita é feita por meio de um fator ou por meio de um número alvo de porções. No fim, as conversões são sempre destinadas a acabar em um fator único.
    - Pensei no fato de algumas coisas não serem necessariamente multiplicáveis e, para garantir isso, as multiplicações são feitas unicamente nos itens marcados como "ingredients"
    - Também pensei que aqueles dados que não possuem valor de quantidade são muito provavelmente devido à escolha de serem "à gosto", como temperos. Então, para esta implementação, eles só não são modificados.
    - Depois os dados dos ingredientes são, de forma reduzida (eliminando ruídos e economizando tokens), enviados para a LLM inferir uma veganização.
    - A saída da veganização é verificada no sentido "Houve Ids mexidos? Algum Step foi removido ou trocado?" e outras perguntas de verificação determinística. Como quase todos os ramos deste sistema, a ideia sempre é "infere->verifica com determinismo"
    - Optei por não escrever testes automatizados para todas as fases de Manipulação e output, vide extrapolação de tempo. Os testes finais foram realizados como testes de sistema dando dados reais para a análise. O comportamento pareceu bom.

12. Nos primeiros testes, percebi que os resultados reduziram o número de dados coletados. Identifiquei que isso ocorria devido a uma falha no "Jsonify". A comparação principal exigia que os dados fossem 100% idênticos entre os quotes para evitar alucinações. No primeiro output.json que eu recebi, vi que a distinção entre a quote e o dado original era de uma vírgula. Pensei: então pra que restringir tanto assim? Implementei pelo OpenCode uma modificação que ajusta para "é 'quase' igual?" e então decide se aceita ou não. Acredito que isso recupere dados adequadamente, visto que alucinações maiores é que são o verdadeiro problema, e essas provavelmente cairão no filtro dos "90% strings iguais" que eu adicionei.

13. Depois de todos os ajustes, achei que o resultado ficou bem legal pra algo feito em menos de um dia. Dado o limite de tempo estabelecido no enunciado da tarefa, optei por encerrar a produção por aqui. Não ficou perfeito, mas ficou melhor do que eu imaginei que ficaria, dado todos os percalços no caminho. Testei o resultado com outras 3 receitas (2 em português e uma em inglês), além da Recipe.pdf.

Acho importante destacar que algumas coisas eu decidi não investir tanto tempo, como a estruturação das tools na coleta dos dados. A LLM não foi incentivada a inferir coisas, o que leva a uma certa dificuldade em detectar ferramentas que estão implicitamente na cena. Entretanto, uma esmagadora maioria dos warnings o modelo conseguiu detectar corretamente o erro e esse histórico ficou disponível pra ver que "olha só, eu fiquei em dúvida se era isso ou aquilo ou mesmo algo além das duas opções da busca". Usar modelos mais potentes em visão computacional ajudaria a reconhecer melhor as imagens. Talvez um dos gargalos seja justamente esse: limitação do Qwen, além do próprio tempo de produção deste software.

A produção desse sistema levou aproximadamente 9 horas e meia.

Testes automatizados foram essencialmente auto-gerados para poupar tempo e guiar a implementação do opencode.

14. Se eu pudesse investir mais tempo, eu provavelmente gastaria ele adicionando passos de verificação a mais na fase de extração, melhorando o resultado da coleta, e testaria outros modelos de LLM para comparar os resultados. Acho que o output poderia ter uma formatação melhor e eu confesso que, apesar de funcional, deve existir solução melhor do que a que eu usei no Extract. Na etapa de Manipulação, pegaria os principais produtos de origem animal e faria conversão direta para produtos veganos reais (me deparei no modelo sugerindo "ovo de aveia" no lugar de ovo, e isso não é algo comum no Brasil, pelo menos, se existir), enviando essas sugestões no prompt de veganização. 