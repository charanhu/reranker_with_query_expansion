import os
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import (
    WatsonxLLM,
)

# from genai.model import Credentials
# from genai.schemas import GenerateParams
# from genai.extensions.langchain import LangChainInterface


load_dotenv()

credentials = {
    "url": os.getenv("GA_GENAI_URL"),
    "apikey": os.getenv("GA_GENAI_KEY"),
}
### Defining the project id
import os

try:
    project_id = os.environ["GA_PROJECT_ID"]
except KeyError:
    raise KeyError(
        "Please set the environment variable PROJECT_ID to your Watson Machine Learning project id."
    )

# ['FLAN_T5_XXL', 'FLAN_UL2', 'MT0_XXL', 'GPT_NEOX', 'MPT_7B_INSTRUCT2', 'STARCODER', 'LLAMA_2_70B_CHAT', 'LLAMA_2_13B_CHAT', 'GRANITE_13B_INSTRUCT', 'GRANITE_13B_CHAT']
## Foundation Models on `watsonx.ai`
LLAMA_2_70B_CHAT = ModelTypes.LLAMA_2_70B_CHAT

### Defining the model parameters
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.MIN_NEW_TOKENS: 1,
    # GenParams.RANDOM_SEED: 42,
    GenParams.TEMPERATURE: 0.0,
    # GenParams.TOP_K: 50,
    # GenParams.TOP_P: 1,
    GenParams.REPETITION_PENALTY: 1.1,
    # GenParams.STOP_SEQUENCES: ["\n\n"],
}


### Initialize the model
llama_2_70b_chat_1 = Model(
    model_id=LLAMA_2_70B_CHAT,
    credentials=credentials,
    project_id=project_id,
    params=parameters,
)



## LangChain integration
llama_2_70b_chat_llm_1 = WatsonxLLM(model=llama_2_70b_chat_1)


# llm = llama_2_70b_chat.to_langchain()

# llama_2_70b_chat_llm.dict()

llm_1 = llama_2_70b_chat_llm_1

