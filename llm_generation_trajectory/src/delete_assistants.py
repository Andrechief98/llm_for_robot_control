from openai import OpenAI


client = OpenAI()

from openai import OpenAI
client = OpenAI()


assistants_to_delete = [
    "asst_bcMq2ohaLHiKDUZclNj5GEjc"
    ]

for assistant in assistants_to_delete:
    response = client.beta.assistants.delete(assistant)
    print(response)



