import random
import re
import subprocess
import shlex
import time
from urllib.parse import quote_plus
from mcp.server.fastmcp import FastMCP
from langchain_huggingface import HuggingFaceEmbeddings
from audit import log
from dotenv import load_dotenv
import os

from rag import find_similar_content

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

def search_with_lynx(url: str, timeout: int = 60) -> str:
    try:
        command = [
            "docker",
            "run",
            "--rm",
            "alpine/lynx",
            "-display_charset=UTF-8",
            "-dump",
            url
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            timeout=timeout
        )

        stdout_text = result.stdout.decode('utf-8', errors='replace')
        return stdout_text

    except FileNotFoundError:
        error_message = "Error: 'docker' command not found. Is Docker installed and in your system's PATH?"
        log(error_message)
        return error_message

    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode('utf-8', errors='replace').strip()
        safe_command_str = ' '.join(shlex.quote(arg) for arg in command)
        error_message = (
            f"Error: Docker command failed with exit code {e.returncode}.\n"
            f"Command: {safe_command_str}\n"
            f"Stderr: {stderr_text}"
        )
        log(error_message)
        return error_message

    except subprocess.TimeoutExpired as e:
        stderr_text = ""
        if e.stderr:
            stderr_text = e.stderr.decode('utf-8', errors='replace').strip()
        safe_command_str = ' '.join(shlex.quote(arg) for arg in command)
        error_message = (
             f"Error: Command timed out after {timeout} seconds.\n"
             f"Command: {safe_command_str}\n"
             f"Stderr so far: {stderr_text or 'N/A'}"
        )
        log(error_message)
        return error_message

    except Exception as e:
        command_str = "docker run ..."
        try:
           command_str = ' '.join(shlex.quote(arg) for arg in command)
        except NameError:
           pass
        error_message = f"Error: An unexpected error occurred while trying to run '{command_str}': {e}"
        log(error_message)
        return error_message



def parse_search_output_to_dict(text: str) -> dict[int, dict[str, str | None]]:
    if not text or not isinstance(text, str):
        return {}

    try:
        ref_marker = "\nReferences\n"
        split_pattern = re.compile(r'\n\s*References\s*\n', re.IGNORECASE)
        parts = split_pattern.split(text, 1)

        if len(parts) != 2:
            log("Warning: 'References' section delimiter not found clearly. Parsing might be incomplete.")
            content_part = text
            references_part = ""
        else:
            content_part = parts[0]
            references_part = parts[1]

    except Exception as e:
        log(f"Error splitting content and references: {e}")
        content_part = text
        references_part = ""

    url_map = {}
    if references_part:
        reference_lines = references_part.strip().split('\n')
        for line in reference_lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^(\d+)\.\s+(https?://\S+)', line)
            if match:
                try:
                    num = int(match.group(1))
                    url = match.group(2)
                    url_map[num] = url
                except ValueError:
                    log(f"Warning: Could not parse reference number in line: {line}")
            else:
                log(f"Warning: Skipping malformed reference line: {line}")

    matches = list(re.finditer(r'\[(\d+)\]', content_part))
    results: dict[int, dict[str, str | None]] = {}

    for i, current_match in enumerate(matches):
        try:
            num = int(current_match.group(1))
            start_pos = current_match.end()

            if i + 1 < len(matches):
                end_pos = matches[i+1].start()
            else:
                end_pos = len(content_part)

            description_raw = content_part[start_pos:end_pos]
            description_clean = re.sub(r'\s+', ' ', description_raw).strip()

            if description_clean:
                results[num] = {
                    'description': description_clean,
                    'url': url_map.get(num)
                }

        except ValueError:
            log(f"Warning: Could not parse number from marker: {current_match.group(0)}")
        except Exception as e:
            log(f"Error processing match {current_match.group(0)}: {e}")

    return results


mcp = FastMCP("Web search")

@mcp.tool()
def search_web(search_query: str) -> str:
        """
        Search the Web
        arg: search term
        """
        log(f"Searching for: '{search_query}' using lynx in Docker...")

        if not isinstance(search_query, str):
            search_query = str(search_query)
        encoded_query = quote_plus(search_query)
        url = f"{os.getenv("SEARCH_ENGINE")}?q={encoded_query}"

        results = search_with_lynx(url)
        output = parse_search_output_to_dict(results)

        lim = 3
        doc = ''

        for i in output:
            if i>12 and doc=='':
                return "Relay to the user that there are no relavent results, DO NOT use other tools"
            elif i>4:
                if lim>0:
                    log(f"search result {i}")
                    log(output[i])
                    try:
                        random_sleep_duration = random.uniform(0.0, 0.5)
                        time.sleep(random_sleep_duration)

                        log(output[i]['url'])
                        doc += f"{search_with_lynx(output[i]['url'])}\n"
                        #log(doc)
                        lim -= 1
                    except Exception as e:
                        log(f"web_search: Error {e}")
            log("one iteration")
        log("done")
        most_relevant = find_similar_content(search_query, doc, embedding=embedding)

        log(f"most relevant search result: {most_relevant}")
        return most_relevant

if __name__ == "__main__":
        mcp.run(transport="stdio")
