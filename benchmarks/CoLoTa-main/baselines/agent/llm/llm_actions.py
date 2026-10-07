from utils.llm.chain import Chain



class GetMonolithicProof(Chain):
    def __init__(self, llm_name, mode,prompts_dir):
        prompt_f = f"{prompts_dir}/GetMonolithicProof-{mode}.yaml"

        super().__init__(prompt_f, llm_name)
    
    def parse_response(self, input_response):
        input = input_response.lower()
        start_marker = "therefore,"
        end_marker = "<|eot_"
        start = input.find(start_marker)
        end = input.find(end_marker)
        if start == -1 and end == -1:
            response = input
        else:
            response = input[start:end]

        if 'false' in response.lower() and 'true' not in response.lower():
            final_answer = 'false'
        elif 'true' in response.lower() and 'false' not in response.lower():
            final_answer = 'true'
        else:
            final_answer = 'misformatted answer'
        return final_answer, input_response
    



class GetFewshotProof(Chain):
    def __init__(self, llm_name, mode, prompts_dir):

        prompt_f = f"{prompts_dir}/GetFewshotProof-{mode}.yaml"
        super().__init__(prompt_f, llm_name)
        
    
    def parse_response(self, input_response):
        input = input_response.lower()
        start_marker = "therefore,"
        end_marker = "<|eot_"
        start = input.find(start_marker)
        end = input.find(end_marker)
        if start == -1 and end == -1:
            response = input
        else:
            response = input[start:end]

        if 'false' in response.lower() and 'true' not in response.lower():
            final_answer = 'false'
        elif 'true' in response.lower() and 'false' not in response.lower():
            final_answer = 'true'
        else:
            final_answer = 'misformatted answer'
        return final_answer, input_response
    