import os
import json
import requests
from typing import Dict, Any
from bs4 import BeautifulSoup
from openai import OpenAI
from urllib.parse import urlparse

class JobDescriptionExtractor:
    def __init__(self, api_key: str):
        """
        Initialize the job description extractor with OpenAI client
        
        :param api_key: OpenAI API key for processing job descriptions
        """
        self.client = OpenAI(api_key=api_key)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extract_job_details(self, job_url: str) -> Dict[str, Any]:
        """
        Extract job details from a given URL
        
        :param job_url: URL of the job posting
        :return: Structured job details dictionary
        """
        # Fetch webpage content
        try:
            response = requests.get(job_url, headers=self.headers)
            response.raise_for_status()
            html_content = response.text
        except requests.RequestException as e:
            raise ValueError(f"Error fetching job URL: {e}")

        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract raw text content
        page_text = self._extract_text(soup)

        # Use OpenAI to structure job details
        system_message = """
        You are an expert job description analyzer. Extract and structure job details from the provided text.
        Create a comprehensive JSON with the following sections completely comprehensive,detailed  and dynamic based on provided job details:
        - Job Title
        - Company
        - Location
        - Job Type (Full-time, Part-time, Contract, etc.)
        - Salary Range (if available)
        - Job Description
        - Key Responsibilities
        - Required Qualifications
        - Preferred Qualifications
        - Skills
        - Industry
        - Department
        - Application Instructions
        
        Be precise, extract all available information, and maintain a professional format.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Analyze and structure the job details:\n\n{page_text}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            parsed_job_details = json.loads(response.choices[0].message.content)
            
            # Add source URL
            parsed_job_details['source_url'] = job_url
            
            return parsed_job_details

        except Exception as e:
            raise Exception(f"Error parsing job description: {str(e)}")

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract meaningful text from BeautifulSoup object
        
        :param soup: BeautifulSoup parsed HTML
        :return: Extracted text content
        """
        # Remove script, style, and navigation elements
        for script in soup(["script", "style", "nav"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text

    def save_job_details(self, job_details: Dict[str, Any], output_path: str):
        """
        Save job details to a JSON file
        
        :param job_details: Structured job details dictionary
        :param output_path: Path to save the JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(job_details, f, indent=2, ensure_ascii=False)

def main():
    # Example usage
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    extractor = JobDescriptionExtractor(api_key)
    
    try:
        # Example job URL (replace with actual URL)
        job_url = "job_url"
        output_file = "job_details.json"
        
        job_details = extractor.extract_job_details(job_url)
        extractor.save_job_details(job_details, output_file)
        
        print(f"Job details extracted successfully! Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()