# mtg_card_creator

This is an experiment aimed at training two LLMs on the Magic the Gathering card game: a text generation model for the card text (starting from its name) and one model for the cards' art.

## Dataset and Training, text model

The text data was taken from https://mtgjson.com/downloads/all-files/

The model was fine-tuned from gpt2, with the objective of generating all of the card's textual information, starting form the name.
This is particularly difficult, because obviously the card's rules do not depend on the name, yet some rules can still be inferred: and elf would probably have the color green, a goblin wouldn't fly and so on.

The training has been done for 40 epoch on a M3 chip. More info in the notebook in /src.


## Dataset and Training, art model

In order to create the image dataset, I have used https://github.com/Investigamer/mtg-art-downloader.

Then I fine-tuned the model of https://huggingface.co/volrath50/fantasy-card-diffusion/tree/main, which was already well-performing.

## Card creation

Implementing a proper card generator that was filling a card outline with the models' info was not part of the experiment. 
I simply made a code that connects through chrome (using chromedriver) to a website named mtgcardmaker, fills the forms and downloads the filled card outline.
Then I pasted the art on the outline using Pillow. Some results are in /cards_output/
