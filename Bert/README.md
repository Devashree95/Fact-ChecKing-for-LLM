# BERT:  
1. Update config.py file with your supabase and openai credentials and table details.
2. Save the PDF files in input directory
3. In order to create a new table in Supabase to store vector data, please enable the vector extension in the Supabase database:  
   ![image](https://github.com/Devashree95/Fact-ChecKing-for-LLM/assets/122653285/ee23d8f9-b226-4eaf-9bcd-b2da0c12395c)
4. Run the create_table.sql script in Supabase SQL editor to create table in supabase.
5. Run load_data.py script to convert the data into vectors and store the same in Supabase.  
6. The 'test_files' folder contains file containing 100 sentences along with their expected classification tag. This folder also contains the results of running these 100 setences for fact checking. 
7. Run evaluate.py file to test sentences of fact checking. You can add your own sentences from knowledge siurce to fact check.
