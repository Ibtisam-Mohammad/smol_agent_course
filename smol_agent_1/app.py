import re
from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from tools.web_search import DuckDuckGoSearchTool
from tools.visit_webpage import VisitWebpageTool

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def searcher(topic: str, number_of_links_to_search: int) -> list:
    """A tool searches the web for the papers published in the top journals since 2020 and their urls.
    Args:
        topic: topic name
        number_of_links_to_search: an integer specifying the number of results to return
    """
    response = DuckDuckGoSearchTool().forward(f"papers published in the journals since 2020 {topic}")
    urls = re.findall(r'\((http[s]?://[^\)]+)\)', response)
    unique_urls = list(set(urls))
    return unique_urls[:number_of_links_to_search]  # Return the specified number of unique URLs

@tool
def web_search(url: str) -> str:
    """A tool searches the web based on the given url.
    Args:
        url: url of the webpage to search
    """
    webpage_content = ""
    if url:
        webpage_content += VisitWebpageTool().forward(url) + "\n"
        print("++" * 10, webpage_content)
    else:
        webpage_content = "No URLs found in the search results."
    return webpage_content


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, web_search, searcher], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="MyResearchAgent",
    description="This agent assists with web interactions.",
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()