# Fact-ChecKing-for-LLM

# Introduction:<br>
Large Language Models (LLMs) are increasingly being integrated into educational system. They offer scalable and adptive learning solutions. But some times, LLMs can generate inaccurate and misleading information. This is particularly concerning for educational settings. We have implemented a fact checking system for educational content using RAG(Retrival Augmented Generation) technique. Our project follows following methodology:

![image](https://github.com/Devashree95/Fact-ChecKing-for-LLM/assets/122653285/23b106d7-3770-47e3-8bcb-7e415022963c)

This project is divided in 2 parts:  
1. Doc2Vec
2. BERT

Same data can be used as knowledge source for both of these parts.  

# 1) Data load:<br>
We used the 9 books stoed at following location and a web URL as our knowledge source. You can download these books in order to run the data load script:
Data: https://1drv.ms/f/s!AgfTSSv8-frFgbB_m6Pwzc8kZzmPnA?e=azYllF  
Web URL: https://www.geeksforgeeks.org/machine-learning/  

# Prerequisites:  
1. OpenAI access key  
2. Supabase account (To store the vector data)  

Please follow following instructions to run the code:  
# Notebooks:  
The Doc2Vec.ipynb and bert.ipynb notebooks uploaded in Notebooks directory can be used to go through the complete process from data load to evaluation. These notebooks contain most of the main functionality of our project and have been created to simplify the code run.  

# Doc2Vec:  
# BERT:  
1. Update config.py file with your supabase and openai credentials and table details.  
2. Run the create_table.sql script to create table in supabase.  
3. Run load_data.py script to convert the data into vectors and store the same in Supabase.  
4. The 'test_files' folder contains file containing 100 sentences along with their expected classification tag. This folder also contains the results of running these 100 setences for fact checking. 
5. Run evaluate.py file to test sentences of fact checking. You can add your own sentences from knowledge siurce to fact check.
   




