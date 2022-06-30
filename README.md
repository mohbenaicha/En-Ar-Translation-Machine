# En-Ar-Translation-Machine

This is a Python package and API I built for translating English sentences to Arabic. The package and API are kept in this single repo to facilitate access to them; otherwise, the package is downloadable through `pip install --extra-index-url https://pypi.fury.io/mohbenaicha/ en_ar_translator==0.0.21` for the latest version and the API is accessible at: https://en-ar-translation-app.herokuapp.com/.


## Test out the model
- The API's accessible through its graphical UI: https://en-ar-translation-app.herokuapp.com/ (the app runs on a single minimal cloud instance and may be sleeping so it may take some time to load)

#### Using the API's graphical UI:
- Open the 'proceed' link in a new tab and keep the existing tab to follow the rest of the instructions.
- Expand the 'POST' request section
- Find the 'try it out' button.
- Edit the 'Request body' input box to your query for translation, then hit 'execute'.
- Scroll down further (ignore the cURL request template following it)
- Check the translation under the reponse body.


#### Using the API through cURL:

```
curl -X 'POST' \
  'https://en-ar-translation-app.herokuapp.com/api/v1/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": [
    {
      "input_sentence": "Translate this sentence for me, please?"
    }
  ]
}'
```

## Alternatively, make a prediction manually:
- install the package `pip install --extra-index-url https://pypi.fury.io/mohbenaicha/ en_ar_translator==0.0.21` (ideally, in a new virtual environment)
- launch python in the environment used to install en_ar_translator
- import the package and translate:

```
from translator_model import translate
print(translate.make_translation(input_data=["Translate this sentence for me."]))
```

## Or, run the API locally: 
  - create a new folder locally, cd into the folder and `git init` it, then clone the current GitHub repo 
  
      `git clone https://github.com/mohbenaicha/En-Ar-Translation-Machine`
      
  - cd into the /translator-api directory, make sure you have `tox` installed (`pip install tox`), 
  - run `tox -e run` to host the API
  - copy+paste the url provided by uvicorn into a browser.
  - follow the steps on using the API above 'Using the API's graphical UI'


## Appendix: Notes and Disclaimers:

- Note 1: status code 200 means the post request was valid and should yield a translation within the response body.
- Note 2: This tanslator was trained on 10,000 En-Ar sentence pairs and so it is not the most sophisticated. It can handle short sentences. This translator is being further trained on more En-Ar sentence pairs as they are collected.
