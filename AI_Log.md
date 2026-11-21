# AI Log

This log is structured indicating the path adopted in crafting the code using OpenCode as an AI Agent. It was originally written in Portuguese and subsequently converted to English. The choice of the original language is due to the developer's greater familiarity with their native tongue, allowing for better articulation in the choice of words.

1. I started the process by selecting the technologies I could use, searching for those that would make sense within the scope of the cookbook. First, I read about OCR and how it is properly used nowadays. Old technology used visual patterns to identify letters and text in images, which did not have the accuracy we have today using LLMs or even simpler structures like Tesseract (pure OCR).
My choice was to work in a hybrid way, not solely with today's very strong LLMs. And the reason? Well, in the first version conceived for this software, I realized that I could perform several subsequent verifications after a very good network, like Qwen2.5, extracted all the text from a PDF or image. However, if Qwen made a mistake right at the start, all subsequent verifications would be in vain (garbage in, garbage out). The solution then? Double the collection. Except if I placed two parallel LLMs, thinking about the idea of running everything locally, it would become something very expensive in terms of hardware to maintain, considering usage on an entry-level PC.

I considered the idea of plugging in an API from one of those large networks out there, like Gemini or GPT, but I drew a parallel with data security, besides the fact that, for most cases, this would be a paid solution. What if the recipe were confidential? What if the transcription were of a recipe from a century-old book that holds a family secret? Better to do this locally.

And of course, no less importantly, the chosen language was Python!

2. With the initial extraction idea, I talked about the architecture with the Big Pickle model, from OpenCode, chosen as the best free Code Agent option that I managed to adapt to. The idea was to discuss the implementation to start with the first block of the system: ingest.py
During this conversation, I came across some interesting points that I reflected upon:
- If the input is a PDF and this PDF is partially built through images, it is prudent to convert the entire PDF page into full images so that if there is any important information within the images themselves, amidst the plain text, it is not lost.
- However, if there is plain text in the PDF, it is interesting to use it as a verification for new extractions, that is, if I have part of the PDF scanned and part typed, I do not want to lose the deterministic (typed) part due to hallucination errors. It makes more sense to compare the text data and that of a detection by Qwen. If Qwen diverges, the text data is assumed. Besides the fact that it is possible to give the model the plain text for pre-detection reference.
  - If I have the images of the PDF pages, it is smart to modify the contrast through OpenCV, so as to make words sharper to facilitate the model's detection.
  - On this point, I discussed with the model and decided to abort this idea. The suggestion that came to me from it, associated with OpenCV, was to use EXIF methods and light CLAHE to increase sharpness. However, I started thinking that this initial project was getting too deep and preferred to keep it without the aid of OpenCV.
- And of course, if I already got the input in images, I just process it. And also, if I already received the input as a .txt or plain text, no processing is necessary.

The techStack of this phase was then: PyMuPDF and OpenCV libraries

Important decisions on some details:
- If reading a PDF page fails, what happens? I opted to discard the page and display partial results with a warning. Even though this is not a complete recipe, it still seems to have some meaning.
- When the input is a set of images, what happens? I respect the order given in the input and read them with OpenCV.

3. With the initial architecture decisions of ingest.py, I decided to move forward to building extract.py. This second file is responsible for calling the LLM and delivering the IngestResult output package for extraction.
I started pondering the idea of using an OCR to generate supporting text in the LLM's inference. However, I gave up on the idea of sending the inference alongside the original content of the input to the LLM for fear of anchoring the inference. I considered masking the data with number or name tags or other data, but I considered this hypothesis too weak.

After some reflection, I decided to go with the approach where the LLM is the OCR's corrector. I picked RapidOCR as an attachable library to the project (chosen for its easy download via pip). I also considered using Tesseract, but eliminated it because it required downloading binaries externally (the choice was more for a matter of practicality than necessarily for efficiency or quality of the result). So the logic became the following:
- OCR and LLM infer on the text.
- LLM returns a JSON with the properly structured data.
- A comparison process begins between the output of the OCR and the LLM.
- If the two outputs are equal, it is a match, and that data is accepted.
- If they are distinct, the OCR's bbox is used to extract the image from the original file and this information is sent again to the OCR and the LLM in isolation. If the result is the same, match. If different, it is called unresolved and generates a warning.
- If a segment ends up as unresolved, the value with the better score (whether the OCR's or the LLM's) is decided upon, just to proceed with the remaining steps. Only from here on out it is already considered that the result is partial, given that it failed.

I thought of this perspective to serve as a kind of brake. I understood that the OCR would be able to limit the high capability of the LLM, so as to prevent it from deducing everything it barely understands (or hallucinating in the process). The double verification (tie-breaker criterion), however, is a very large hardware bottleneck of this approach. Among all options, I associate reading the image as the most important part of the entire software, given that this is where there is a high possibility of the input being bad (garbage in, garbage out). If the input is bad, all the rest of the software is essentially useless.

I believe there must exist more optimized approaches for solving this problem, but among what I researched and talked about with opencode, it was what sounded most reasonable within the scope of the project. Better results would require more development time and I considered the safety of this approach sufficient for the coding challenge.

4. I had truncation problems in the model's output. I opted to reduce the input image size even further to leave more tokens for the output. Another approach would be to change Ollama's limit by giving more tokens and context, but I decided to go down the image reduction path instead.

5. I had more problems associated with the formats returned by the model. The schema was not fitting and data ended up being discarded in the process. This caused the recipe to always end up incomplete. Adjustments associated with the prompt were made, clarifying formats

6. I noticed that the preliminary results were not good enough and decided to change the extraction reasoning. I deleted the preliminary modifications and made an initial multi-file model for Extract. The folder appeared and ended up on Git right after.

7. I redid the basic engineering of the architecture, emphasizing exactly what was described in item 3 of this log, trying for better results. I realized that some of my statements in the model's chat were vague and it inferred things in the architecture that went beyond the specification I developed. That is why discarding the original code was the right decision. I redid it more calmly and decided to put the entire specification in SPEC.md and reset the chat for this implementation stage. OpenCode read the restructured SPEC.md and applied the additional fixes to the new file structure.... And it went wrong.

8. I decided to change everything again, starting from scratch and excluding the idea of using OCR. I compared the two extractions and this was the difference:

LLM:

189 Will man den Braten nach dem 1. Tag oder 2. Tag gebrauchen, so giept man das Essig- wasser heiß darüber. Soll der Braten erst nach 8 Tagen zubereitet werden, so giept man es kalt darauf. Zum Anbraten den Speck auslassen, Butter dazugeben, dann gibt man den abgetrockne- ten Braten ins heiße Fett. Wenn er von al- len Seiten knusprig angebraten ist, giept man etwas Essigwasser hinzu. Auch kann man einige Äpfel, Möhren und Wacholderbeeren in den Topf geben. Man läßt den Braten langsam mit den Zutaten gar- brötscheln. Für die Soße schwitzt man den Bratenfond mit Mehl an, füllt mit Essigwasser und Rot- wein auf, gibt dann gequollene Korinthen u. Rosinen hinzu und soviel Zuckcr, daß die Soße einen säuerlich-süßen Geschmack bekommt. Man legt den Braten, wenn er noch nicht ge- schnitten ist, eine Stunde vor dem Anrich- ten in die Soße, damit er den Geschmack derselben bekommt. Rotwein Korinthen

OCR:

189 Willman den Braten nach h dem 1.Tag oder es'kall darauy. auslassen,BuHer tenBraten ins heiBe fet. wenn er von al- len Serlen Rnusprig qn gebraten ist, giepk man elwas Essigwa sser yinzu brotscheln. fir dieSo/3e Schwitet nan den Bratentond mi Hehl qn tilltnit tssigwasserundRot weinquy qibt dann gequollene Korinthen u. Rosinen' hinee und Soviel. Zudker, da/3die Soybeeinen - sifBen Geschmack bekommt. ManlegtdenBraten wenn er noch hnichtge SchniHenist eine Stunde vor dem Anrid ten in die Sopbe , damib er den geschmack derselbenbekommt. ROEwg Kori then

The quality obtained in the elaboration with pure LLM was much better than that of the OCR... This means that the OCR does not brake the LLM... It simply destroys the quality of the LLM's inference. I will start again without OCR and with another detection method.

9. The new method was the following:
- Now we use only the LLM for all inference, but with two detection layers.
  - The first layer is a general pass extracting the text of the entire document
  - The second layer consists of breaking each image into 2/3 portions: a crop of the image containing 2/3 of it from top to bottom and another crop of the image containing 2/3 of it from bottom to top.
  - Deterministically, through pure Python, the two data sets are compared. If there were small discrepancies between some of the data, a warning is emitted for each of the discrepancies. Equal data is accepted or it is simply assumed that the collection from the cropped images is the correct answer.
  - A prompt with the final text is sent to the same LLM requesting conversion to JSON. The engineering here is merely prompt-based and was the simplest solution I could think of.
  - After returning the JSON, duplicates that are in the same step are checked, as well as whether the quote is contained in the input text before the JSON (ensuring that no hallucinations occurred)
  - Returns this JSON to the Manipulation stage.

    The first tests were quite promising and there were indeed significant distinctions between the cropped results and the full-page results. I assumed this version as the ideal one.
    The downside was the inference time. I felt that by running the LLM multiple times in search of results, the processing time increased a lot. Since my hardware is limited, it was a burden of this implementation.

10. After the stress and time of approximately 4 hours of extraction development (given the various architectural changes adopted, not all explicitly referenced in this log), I moved on to the Manipulation phase. I believed the next two stages would demand much less time from me than the extraction stage, since more deterministic verifications would make sense. I took advantage of the flow and already implemented the output alongside it.

Up to this moment, I had been using a dummy "main" to test the code in addition to automated unit tests and OpenCode's own inferences from questions I asked. The implementation also allocated a main.py to facilitate system tests and identification of general bugs. I did all of this somewhat together to close the package neatly and only refine isolated snippets.

The output was very deterministic (so much so that I did not use LLM processing in it). The output is in list format inside a .md. The choice of formatting the .md, given that it would be something purely visual and deterministic, I left in the hands of my agent. I preserved the output.json and warnings.txt outputs to have error traceability and allow any other destination for the data collected in this whole process.

11. Explaining better what I did in the manipulation part, I adopted this sequence:
- I convert the numerical data I received into a purely numerical form (without unicode characters of 1/2 or 1/8 or anything like that)
  - Deterministically I define whether the recipe's scalability is done through a factor or through a target number of servings. In the end, conversions are always destined to end up in a single factor.
  - I thought about the fact that some things are not necessarily multipliable and, to ensure this, multiplications are made solely on items marked as "ingredients"
  - I also thought that those data entries that do not have a quantity value are very likely due to the choice of being "to taste", like seasonings. So, for this implementation, they are simply not modified.
  - Then the ingredient data is, in a reduced form (eliminating noise and saving tokens), sent to the LLM to infer a veganization.
  - The veganization output is verified in the sense of "Were IDs tampered with? Was any Step removed or swapped?" and other deterministic verification questions. Like almost all branches of this system, the idea is always "infer -> verify with determinism"
  - I opted not to write automated tests for all phases of Manipulation and output, given the time overrun. The final tests were performed as system tests giving real data for analysis. The behavior seemed good.

12. In the first tests, I noticed that the results reduced the amount of collected data. I identified that this occurred due to a flaw in "Jsonify". The main comparison required the data to be 100% identical between the quotes to avoid hallucinations. In the first output.json I received, I saw that the distinction between the quote and the original data was a single comma. I thought: then why restrict so much? I implemented through OpenCode a modification that adjusts to "is it 'almost' equal?" and then decides whether to accept it or not. I believe this recovers data adequately, since larger hallucinations are the real problem, and those will probably fall into the "90% equal strings" filter I added.

13. After all the adjustments, I thought the result turned out pretty cool for something done in less than a day. Given the time limit established in the task statement, I opted to wrap up production here. It wasn't perfect, but it turned out better than I imagined it would, given all the setbacks along the way. I tested the result with 3 other recipes (2 in Portuguese and one in English), in addition to Recipe.pdf.

I think it is important to highlight that there are some things I decided not to invest as much time in, such as structuring the tools in data collection. The LLM was not encouraged to infer things, which leads to a certain difficulty in detecting tools that are implicitly in the scene. However, in an overwhelming majority of the warnings, the model managed to correctly detect the error and this history became available to see that "look, I was in doubt whether it was this or that or even something beyond the two search options". Using more powerful models in computer vision would help recognize the images better. Perhaps one of the bottlenecks is precisely this: Qwen's limitation, in addition to the production time of this software itself.

The production of this system took approximately 9 and a half hours.

Automated tests were essentially auto-generated to save time and guide OpenCode's implementation.

14. If I could invest more time, I would probably spend it adding extra verification steps in the extraction phase, improving the collection result, and I would test other LLM models to compare the results. I think the output could have better formatting and I confess that, although functional, there must exist a better solution than the one I used in Extract. In the Manipulation stage, I would take the main animal-origin products and make a direct conversion to real vegan products (I came across the model suggesting "oat egg" instead of egg, and that is not something common in Brazil, at least, if it even exists), sending these suggestions in the veganization prompt.