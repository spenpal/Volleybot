# Volleybot
**A simple chatbot to answer any questions related to volleyball!**

This project was completed as part of the [University of Texas at Dallas' CS 4395 Human Language Technologies curriculum](https://catalog.utdallas.edu/2021/undergraduate/courses/cs4395), respectively taught by [Dr. Karen Mazidi](https://cs.utdallas.edu/people/faculty/karen-mazidi/), during the Spring 2021 semester.

## Objective
- Knowledge Base
    - Create a knowledge base scraped from the web.
- Chatbot
    - Create a chatbot using NLP techniques learned in class. The chatbot should be able to carry on a limited conversation in a particular domain using a knowledge base, and knowledge it learns from the user.
    - A user model is also maintained within the chatbot system. A different user model is saved for each user who converses with the bot. The user model should store the user’s  name, personal information it gathers from the dialog, and the user’s likes and dislikes. Add personalized remarks from the user model to the dialog engine.


## Report
You can read our chatbot design & analysis [here](https://github.com/spenpal/Volleybot/blob/main/chatbot_report.pdf).


## Web Crawling/Scraping Notes
- Web crawler grabs as many specificed links as needed.
- Web scraper scrapes only paragraph elements, to avoid getting header or footer text.
    - The number of the scraped file, in the title, corresponds to its relevant url number, found in *relevant_urls.txt*.
- Before finding important terms, manual parsing through cleaned scraped data is done first, to ensure scraped content is quality.
    - Moderate changes were made to cleaned scraped files, such as removing redundant lines, non-english characters, and empty files, duplicate files. This may be why there may be less cleaned files than scraped files.
    - There are a total of **20** cleaned files used to find important terms and compile a knowledge base out of.
- Knowlege base is in form of a Python dictionary with keywords being an important term from the corpus, and value being a list of sentences that has the important term in it.





