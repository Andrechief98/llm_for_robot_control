from openai import OpenAI


client = OpenAI()

from openai import OpenAI
client = OpenAI()


threads_to_delete = [
    "thread_bfFQ9qzaQ342AMtXp6k4sy0d"
    ]

for thread in threads_to_delete:
    response = client.beta.threads.delete(thread)
    print(response)



