import os
import uuid
from openai import OpenAI
from dotenv import load_dotenv
from agentic_doc.parse import parse
from unstructured.cleaners.core import clean_extra_whitespace

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_pipeline(input_path, output_folder):
    """
    Generalized cleaning pipeline optimized for backend integration.
    Removes hardcoded paths and uses unique IDs to prevent file overwrites.
    """
    # 1. GENERATE UNIQUE ID: Essential for concurrent backend users
    job_id = str(uuid.uuid4())[:8]
    original_name = os.path.basename(input_path).split('.')[0]
    
    # 2. CONVERSION: PDF to Raw Markdown
    print(f"🚀 [Job {job_id}] Starting conversion: {original_name}")
    results = parse([input_path])
    raw_content = results[0].markdown
    
    # 3. CHUNKED REFINEMENT: Handling long proposals safely
    # We clean in chunks to avoid the 30,000 token limit that crashed Vendor B
    print(f"🧹 [Job {job_id}] Refining content in safe chunks...")
    
    # Increase this if you want more context, decrease if you hit rate limits
    chunk_size = 5000 
    content_chunks = [raw_content[i:i+chunk_size] for i in range(0, len(raw_content), chunk_size)]
    refined_parts = []

    for chunk in content_chunks:
        refine_prompt = f"""
        Clean this Markdown segment. Rules:
        - FIX BROKEN WORDS: (e.g., 'cap- stone' -> 'capstone').
        - REMOVE LOGOS/FOOTERS: Strip visual descriptions and repetitive headers.
        - PRESERVE ARABIC: Do not translate or alter technical terms.
        
        Text:
        {chunk}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini", # Use mini for cost and high rate limits
            messages=[{"role": "system", "content": "You are a professional document cleaner."},
                      {"role": "user", "content": refine_prompt}]
        )
        refined_parts.append(response.choices[0].message.content)

    # 4. FINAL POLISH: Combine and clean whitespace
    full_refined_text = "\n".join(refined_parts)
    final_text = clean_extra_whitespace(full_refined_text)

    # 5. SAVE WITH UNIQUE NAME: Prevents overwriting Vendor A with Vendor B
    final_filename = f"{original_name}_{job_id}_cleaned.md"
    final_md_path = os.path.join(output_folder, final_filename)
    
    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    
    return final_md_path

if __name__ == "__main__":
    # Test paths
    TEST_INPUT = r"D:\Capstone_Project_SDAIA\src\data\raw\Vendor C.pdf"
    TEST_OUTPUT = r"D:\Capstone_Project_SDAIA\src\data\processed"
    os.makedirs(TEST_OUTPUT, exist_ok=True)
    
    try:
        path = run_pipeline(TEST_INPUT, TEST_OUTPUT)
        print(f"✨ Success! Backend-ready file saved at: {path}")
    except Exception as e:
        print(f"❌ Error: {e}")