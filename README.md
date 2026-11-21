# AI Powered Recipe Parser

## How to use it

Using this system is quite simple, requiring only a properly configured Python environment and a properly performed Ollama installation or an API key from some provider. I suggest using a Conda environment and inserting the necessary libraries.

To install Ollama, access https://ollama.com/ and install the app. The recommended model is Qwen2.5-VL-7b. Install it using
    
    ollama pull qwen2.5vl:7b

If you prefer another provider, change the specifications in the .env following the .env.example

Create a conda environment with Python version 3.11 and install the requirements available in the `requirements.txt` file

---

After that, all operation is via the command line. The system, by default, searches for Recipe.pdf if no data is passed to it as an argument.

The typical use of the system will follow this command line:

    python main.py --servings 6 --base-servings 1 --vegan

or

    python main.py --scale 3 --base-servings 2 --vegan

Omitting the `--vegan` parameter causes only the number of servings to be modified. Omitting serving parameters and `--vegan` causes the software to only convert the input to .md through the main routine.

If the file to be read is not `Recipe.pdf`, pass the path as an argument (first parameter right after main.py), as in the example:

    python main.py MyCookbook.jpg --servings 6 --base-servings 1 --vegan

The expected output will be created in the `results/` folder, containing `output.json`, `warnings.txt`, and `recipe.md`. The first file consists of the structure of what was collected, so as to allow use in another location beyond the system. The second is the list of moments in which possible failures on the part of the LLM were recorded. And the last file consists of the formatted step-by-step output of the extracted recipe.

## Architecture

The program is distributed into 4 branches: ingest, extract, manipulate, and output.

Ingest works under the following premises:
- PDFs, .txt, and images are the only accepted inputs
- Locked or broken data is rejected and warnings are generated
- If we are talking about a PDF, its plain text will be checked (if any) and it will be used.
- If the PDF does not have text (or has partial text), its pages are converted into images by PyMuPDF
- The images and/or text are sent to the Extraction stage.

Extraction works under the following premises:
- For images, it is necessary to extract the text from them through the LLM
- Then the existing full text is requested
- After that, each image is cropped into two halves: one with 2/3 of the image from top to bottom and another with 2/3 of the image from bottom to top.
- These two images are used to extract the full text again and everything is concatenated
- The two extractions (full or in two steps) are compared and checked for discrepancies
- If there are differences, the position is scored and a warning is generated (which will be displayed in the output warnings.txt)
- The polished text is then sent to the same LLM to extract the JSON with ingredients, tools, and steps
- Each ingredient and tool belongs to a step
- Deterministic checks ensure that everything the LLM extracts into JSON is contained in the text polished by the initial double verification (of two detections) and remove duplicates from distinct steps
  - Here lies the main problem of this system: not much polishing was done on this output, so when the model gets lost when assembling an adequate structure in the .md, it is normally due to this stage. At this point, I would improve the prompt and perform new duplicate verifications, ensuring a progressive advance of "Build -> Verify -> Build -> Verify"
- JSON is returned containing everything that matters, including what is reliable and what is not (with this warned)

Manipulate works under the following premises:
- Modifying the number of servings is deterministic. Only the coefficient is calculated (when the number of servings is given in the input instead of it) and multiplied by the value.
- Ingredients without quantity are considered "to taste", so they are not modified
- Veganizing requires the conversion of exclusively animal-origin ingredients and they are sent to the LLM chosen in the .env (TEXT_MODEL) for this purpose
- Verifications in this stage include only barriers so that the model does not invent IDs or modify something that was not requested. If there are discrepancies, warnings are generated.
- Vegan products are selected and meat quantities are deterministically multiplied by 1.5 (this was defined arbitrarily). Derivatives are kept at a 1.0 multiplication (such as eggs or milk).
- Results are sent to the output

Output works under the following premises:
- It is built deterministically, so it does not use an LLM at any moment
- It only structures the output into Markdown.
- It was done in a somewhat rushed way, so it is something I would polish more severely if I had more time.


## Considerations

Given production time limitations, this system is not 100% polished and may present failures especially regarding the recipe's Tools field. After several tests, two things were found (and the first one somewhat obvious): difficult data (especially messy handwritten) brings imprecise results sometimes, model quality is the anchor for the result being good or bad (and thus it was found that Qwen infers texts in the English language better, despite having correctly collected information in German and Portuguese in the tests).

The development flow of this system is briefly documented in the `AI_log.md` file, containing an initial flow of the software production. Some critical decision-making and even more than one architecture of the Extraction block are described there, given the change of direction at a certain moment in production.

This system was built using the OpenCode Big Pickle Agent and some discussions of results with Gemini Hydrogen.

The documentation texts of this system were originally written in Portuguese and subsequently translated automatically (and reviewed by human eyes).