import rospy
from std_srvs.srv import Trigger
from OpenAI_interface.srv import gptCall, gptCallRequest  


class GPTClient:
    def __init__(self):
        rospy.init_node('gpt_client')
        self.service_name = 'gptCall'
        rospy.wait_for_service(self.service_name)
        self.gpt_service = rospy.ServiceProxy(self.service_name, gptCall)

    def call_service(self, prompt, model):
        try:
            request = gptCallRequest()
            request.prompt = prompt
            request.model = model
            response = self.gpt_service(request)
            return response.result  # Adatta al tipo di risposta effettivo
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None
        

if __name__ == "__main__":
    client = GPTClient()
    prompt = "Genera un report sulla sicurezza"
    model = "gpt-4o"
    result = client.call_service(prompt, model)
    if result:
        rospy.loginfo(f"Risultato del servizio: {result}")
