import os
from typing import Dict, Any
import json
from openai import OpenAI
import PyPDF2
import docx
from pathlib import Path

class ResumeParser:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        

    def read_file(self, file_path: str) -> str:
        """
        Read content from different file formats (PDF, DOCX, TXT)
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self._read_pdf(file_path)
        elif file_extension == '.docx':
            return self._read_docx(file_path)
        elif file_extension == '.txt':
            return self._read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _read_pdf(self, file_path: str) -> str:
        """Read content from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text.strip()

    def _read_docx(self, file_path: str) -> str:
        """Read content from DOCX file"""
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs]).strip()

    def _read_txt(self, file_path: str) -> str:
        """Read content from TXT file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    

    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """
        Parse resume and return structured JSON data
        """
        # Read the content
        content = self.read_file(file_path)
        
        
        

        # Create system message that instructs how to parse the resume
        system_message = """
        You are a resume parsing expert. Analyze the provided resume and extract information into a structured JSON format.
        Include the following sections if present:
        - Personal Information (name, contact, location)
        - Summary/Objective
        - Work Experience (with company names, dates, positions, and responsibilities)
        - Education (with institutions, degrees, dates, and relevant details)
        - Skills (technical, soft skills, languages)
        - Certifications
        - Projects
        - Achievements
        - Publications
        - Volunteer Work
        - Additional Information
        
        Create sections dynamically based on the content available in the resume.
        Ensure dates are formatted consistently (YYYY-MM format).
        Include all relevant details while maintaining accuracy.
        """

        # Make API call to OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Parse the following resume and return a JSON structure:\n\n{content}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            parsed_data = json.loads(response.choices[0].message.content)
            return parsed_data

        except Exception as e:
            raise Exception(f"Error parsing resume: {str(e)}")

    def save_parsed_data(self, parsed_data: Dict[str, Any], output_path: str):
        """
        Save parsed data to a JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, indent=2, ensure_ascii=False)

def main():
    # Example usage
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    parser = ResumeParser(api_key)
    
    # Example error handling and usage
    try:
        # Parse resume
        resume_file = "path/to/resume"  # Change this to your resume file path
        output_file = "parsed_resume.json"
        
        parsed_data = parser.parse_resume(resume_file)
        parser.save_parsed_data(parsed_data, output_file)
        
        print(f"Resume parsed successfully! Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()