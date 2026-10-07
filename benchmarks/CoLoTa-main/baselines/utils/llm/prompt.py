from ruamel.yaml import YAML
from jinja2 import Template


class Prompt:
    def __init__(self, prompt_file, llm_name):
        yaml = YAML()
        self.llm_name = llm_name

        with open(prompt_file, 'r') as f:
            self.prompt_info = yaml.load(f)

        self.input_template = Template(
            self.prompt_info['input_template'], trim_blocks=True, lstrip_blocks=True)

    def __call__(self, **input):
        if "gpt" in self.llm_name.lower() or "llama" in self.llm_name.lower() or "mistral" in self.llm_name.lower() or "mixtral" in self.llm_name.lower():
            messages = []
            messages.append(self._create_system_message())

            try:
                messages += self._create_few_shot_messages()
            except:
                pass

            input_message = {'role': 'user', 'content': self._format_input(input)}
            messages.append(input_message)
        
        elif "gemini" in self.llm_name:
            messages = self._create_gemini_messages(input)

        return messages

    def _create_gemini_messages(self, input):
        try:
            fs = ""
            fs_messages = self._create_few_shot_messages()
            for message in fs_messages:
                if message['role'] == 'user':
                    fs += 'input:' + message['content'] + '\n'
                else:
                    fs += 'output:' + message['content'] + '\n'
        except:
            fs_messages = []
        return [self.prompt_info['system'] + fs, self._format_input(input)]

    def _create_system_message(self):
        return {'role': 'system', 'content': self.prompt_info['system']}

    def _create_few_shot_messages(self):
        messages = []

        for example in self.prompt_info['few_shot']:
            input = self._format_input(example['input'])
            output = example['output']

            user_msg = {'role': 'user', 'content': input}
            assistant_msg = {'role': 'assistant', 'content': output}

            messages += [user_msg, assistant_msg]

        return messages

    def _format_input(self, input):
        return self.input_template.render(**input)
