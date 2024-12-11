import re
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import torch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time
import os
import requests
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from PIL import Image


CARD_INFO = "Singing Mermaid"
TEXT_MODEL = "./models/checkpoint-11200"
PICTURE_PATH = "./card_art/"
PICTURE_MODEL_ID = "volrath50/fantasy-card-diffusion"
PICTURE_SAFETENSORS_PATH = "./models/fantasycarddiffusion_140000.safetensors"
CARDMAKER_URL = "https://www.mtgcardmaker.com/"
CARD_OUTPUT_PATH = "./cards_output/"


# Function to truncate a string at the first occurrence of the pattern
def truncate_at_pattern(dic, pattern):
    text = dic['generated_text']
    match = re.search(pattern, text)
    if match:
        # Get the end position of the match and truncate the string
        return text[:match.end()]
    return text  # Return the original text if no match is found


def create_tokenizer():
    model_name = 'openai-community/gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Function to generate the text of the card
def generate_text(card_info):
    model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL)
    tokenizer = create_tokenizer()
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    pattern = r"\[SEP\] [-*\d]/[-*\d]"
    outputs = generator(
        CARD_INFO, 
        max_length=512, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id
    )
    # # Apply truncation
    card_text = truncate_at_pattern(outputs[0], pattern)

    return card_text


# Function to generate a picture of the card
def generate_picture(card_info):

    # Load the Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        PICTURE_MODEL_ID,
        torch_dtype=torch.float32,  # Use float16 precision for efficiency
        safety_checker=None,        # Disable safety checker if not needed
        use_safetensors=True,        # Ensure safetensors compatibility
        weights_path=PICTURE_SAFETENSORS_PATH
    )

    # Move the pipeline to GPU for faster inference
    pipeline.to("mps")

    # Define your prompt
    prompt = f"mtg card art, {card_info}"

    # Generate an image
    image = pipeline(prompt).images[0]
    card_name = CARD_INFO.split(" SEP ")[0].replace(" ", "_").lower()
    picture_path = PICTURE_PATH + card_name + ".png"
    # Save the generated image
    image.save(picture_path)
    print("Image saved")


# Generate card in MTGCardMaker
def split_card_text(card_text):
    form_data = {
        "name": "",
        "subtype": "",
        "cardtext": "",
        "power": "0",
        "toughness": "0",
    }

    selection_data = {
        "color": "White",
        "cardtype": "Creature",
        "mana_w": "0",
        "mana_u": "0",
        "mana_b": "0",
        "mana_r": "0",
        "mana_g": "0",
        "mana_colorless": "0",

    }
    card_parts = card_text.split(" [SEP] ")

    # print('AAAAAAAAAAA')
    # print(card_parts)

    # Extract the name
    form_data["name"] = card_parts[0]

    # Extract the mana cost
    mana_cost = card_parts[1]
    colorless_mana = re.search(r"\{(\d+)\}", mana_cost)
    if colorless_mana:
        selection_data["mana_colorless"] = colorless_mana.group(1)
    letters = re.findall(r"\{([A-Za-z]+)\}", mana_cost)
    for letter in letters:
        selection_data[f"mana_{letter.lower()}"] = \
            str(int(selection_data[f"mana_{letter.lower()}"]) + 1)
    # Card color
    colors = set(letters)
    if len(colors) == 1:
        if "W" in colors:
            selection_data["color"] = "White"
        elif "U" in colors:
            selection_data["color"] = "Blue"
        elif "B" in colors:
            selection_data["color"] = "Black"
        elif "R" in colors:
            selection_data["color"] = "Red"
        elif "G" in colors:
            selection_data["color"] = "Green"
    elif len(colors) > 1:
        selection_data["color"] = "Gold"

    # Extract the type
    card_type_subtype = card_parts[2].split(" — ")

    form_data["subtype"] = card_type_subtype[1]
    selection_data["cardtype"] = card_type_subtype[0]

    # Extract the card text
    form_data["cardtext"] = card_parts[3] + "\n" + "\n" + "[" + card_parts[4] + "]"
    form_data["cardtext"] = form_data["cardtext"].replace("\\n", " \n ")

    # Extract the power and toughness
    power_toughness = card_parts[5].split("/")

    if power_toughness[0] != "-":
        form_data["power"] = power_toughness[0]

        form_data["toughness"] = power_toughness[1]

    return form_data, selection_data


def generate_card_in_mtg_card_maker(card_text):

    form_data, selection_data = split_card_text(card_text)

    # Automatically manage Chromedriver
    service = Service(ChromeDriverManager().install())

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=service)

    try:
        # Open the webpage
        driver.get(CARDMAKER_URL)

        # Wait for the page to load (adjust time if necessary)
        time.sleep(3)
        # consent button
        WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "fc-button"))
            ).click()
        print("Consent dialog bypassed.")

        # Fill textboxes by their 'name' attributes
        for name, value in form_data.items():
            try: 
                textbox = driver.find_element(By.NAME, name)
                textbox.clear()  # Clear any existing text
                textbox.send_keys(value)
            except Exception as e:
                print(f"Could not find textbox with name '{name}': {e}")

        # Fill selection boxes by their 'name' attributes
        for name, value in selection_data.items():
            try:
                
                selectbox = driver.find_element(By.NAME, name)
                dropdown = Select(selectbox)
                dropdown.select_by_visible_text(value)

            except Exception as e:
                print(f"Could not find selection with name '{name}': {e}")

        # Click the 'Generate' button
        generate_button = driver.find_element(By.ID, "generate")
        generate_button.click()

        # Wait for the image to be generated (adjust time as needed)
        time.sleep(5)

        # Locate the image element
        image_element = driver.find_element(By.XPATH, "//img[@id='card']")
        
        # Get the 'src' attribute of the image
        image_url = image_element.get_attribute("src")

        # Download the image using requests
        response = requests.get(image_url)

        if response.status_code == 200:
            # Save the image locally
            card_name = CARD_INFO.split(" SEP ")[0].replace(" ", "_").lower()
            card_output_path = CARD_OUTPUT_PATH + card_name + ".png"
            image_path = os.path.join(card_output_path)
            with open(image_path, "wb") as file:
                file.write(response.content)
            print(f"Image downloaded successfully: {image_path}")
        else:
            print(
                f"Failed to download image: {response.status_code}"
            )
    finally:
        # Close the WebDriver
        driver.quit()

    # Load the background and pic images
    background = Image.open(CARD_OUTPUT_PATH + card_name + ".png")
    pic = Image.open(PICTURE_PATH + card_name + ".png")

    # Define the target square region
    # (left, upper, right, lower) on the background
    target_region = (37, 69, 363, 309)

    # Resize 'pic' to fit the target region
    target_width = target_region[2] - target_region[0]
    target_height = target_region[3] - target_region[1]
    pic_resized = pic.resize((target_width, target_height))

    # Paste 'pic' onto 'background' at the target region
    background.paste(pic_resized, target_region)

    # Save or show the final image
    background.save(CARD_OUTPUT_PATH + card_name + ".png")


if __name__ == "__main__":

    print('-----------------')
    print('Creating card text ...')
    print('-----------------')

    card_text = generate_text(CARD_INFO)

    print('-----------------')
    print(card_text)
    print('-----------------')
    
    print('-----------------')
    print('Creating card picture ...')
    print('-----------------')
    generate_picture(CARD_INFO)

    print('-----------------')
    print('Creating card ...')
    print('-----------------')
    generate_card_in_mtg_card_maker(card_text)
