"""
  Title: Machine Translation API command line tool for translating WMT testsets
  Original author: Tom Kocmi
"""
import glob
import logging
import os
import time
import traceback
from pathlib import Path

import diskcache as dc
from absl import flags, app
from tqdm import tqdm

from tools.models.litellm_api import litellm_gemini_2_5_flash, litellm_llama_70b
from tools.models.openai import openai_gpt4o
from tools.prompts import get_prompt

SYSTEMS = {
    'GPT-4o': {"call": openai_gpt4o, "prompt": "conversation"},
    'Gemini-2.5-Flash': {"call": litellm_gemini_2_5_flash, "prompt": "conversation"},
    'Llama-3.3-70b': {"call": litellm_llama_70b, "prompt": "conversation"},
    }
 
flags.DEFINE_enum('system', "Gemini-2.5-Flash", SYSTEMS, 'Define the system to use for translation')
flags.DEFINE_bool('no_testsuites', False, 'Remove testsuites in case of limited bandwidth')
flags.DEFINE_bool('override', False, 'Ignore existing files and translate again using cache (delete cache manually if needed)')
flags.DEFINE_string('lp', None, 'Translate only specific language pair')

FLAGS = flags.FLAGS


def remove_tripple_quotes(text):
    # check if there are exactly two occurences of ``` in the text
    if text.count("```") == 2:
        # get only the text inbetween the tripple quotes
        text = text.split("```")[1]

    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    # replace new lines and tabs with spaces
    text = text.replace("\n", " ").replace("\r", "").replace("\t", " ")

    return text

def main(args):
    folder = "wmt_testset_romansh/*.full.*"
    if FLAGS.no_testsuites:
        folder = "wmt_testset_romansh/*.no-testsuites.*"

    systems_dir = Path(__file__).parent.parent / "systems"
    assert systems_dir.exists()
    system_folder = systems_dir / FLAGS.system
    if not os.path.exists(system_folder):
        os.makedirs(system_folder)

    cache = dc.Cache(f'cache/{FLAGS.system}', expire=None, size_limit=int(40e10), cull_limit=0, eviction_policy='none')

    none_counter = 0
    for file in tqdm(glob.glob(folder), "Translating WMT testset", position=0):
        lp = file.split(".")[2]
        if "-" not in lp:
            continue
        if FLAGS.lp is not None and FLAGS.lp != lp:
            logging.info(f"Skipping {lp}")
            continue

        source_language = lp.split("-")[0]
        target_language = lp.split("-")[1]

        target_filename = file.split("/")[-1]
        target_filename = f"{system_folder}/{target_filename}"
        if os.path.exists(target_filename) and not FLAGS.override:
            continue

        # read file into list
        with open(file) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        translated = []
        counter = 0
        for line in tqdm(lines, f"Translating with {FLAGS.system}", leave=False, position=1):
            counter += 1
            if SYSTEMS[FLAGS.system]["prompt"] is not None:
                request = {
                    "prompt": get_prompt(line, source_language, target_language, SYSTEMS[FLAGS.system]["prompt"])
                }
            else:
                request = {
                    'source_language': source_language,
                    'target_language': target_language,
                    'segment': line
                }

            # None represent problem in the translation that was originally skipped
            if request in cache and cache[request] is not None:
                print("Cache hit", flush=True)
                translated.append(cache[request])
            else:
                counter = 0
                while True:
                    counter += 1
                    if counter > 5:
                        time.sleep(1)
                        raise Exception("Too many retries")
                    try:
                        print("Calling API ...", flush=True)
                        translated.append(SYSTEMS[FLAGS.system]["call"](**request))
                        break
                    except Exception as e:     
                        # translated.append(None) 
                        # break                  
                        traceback.print_exc()
                        print(e, flush=True)
                        continue

                cache[request] = translated[-1]

        # if most are None, the system doesn't support the language
        if sum([1 for t in translated if t is None]) > len(translated) / 2:
            logging.info(f"Skipping {lp} as it is not supported by {FLAGS.system}")
            continue

        input_token_count = 0
        output_token_count = 0
        with open(target_filename, "w") as f:
            for line in translated:
                translation = line
                # if line is tupple, then it contains also the token count information
                if isinstance(line, tuple):
                    translation = line[0]
                    # input_token_count += line[1][0]
                    # output_token_count += line[1][1]
                elif line is None:
                    translation = ""
                    none_counter += 1

                if SYSTEMS[FLAGS.system]["prompt"] is not None:
                    translation = remove_tripple_quotes(translation)
                
                f.write(translation + "\n")

        # save token count information into file
        with open(f"{target_filename}.tokens", "w") as f:
            f.write(f"Input tokens: {input_token_count}\n")
            f.write(f"Output tokens: {output_token_count}\n")

    logging.info(f"None counter: {none_counter}")

if __name__ == '__main__':
    app.run(main)
