import asyncio
import websockets
from openai import AsyncOpenAI
import speech_recognition as sr
import numpy as np
from scipy.io import wavfile
import webrtcvad
import wave
import pyaudio
import threading
import queue
import os
import json
import tempfile
import re
from datetime import datetime, timedelta
from faster_whisper import WhisperModel
import pygame
from gtts import gTTS
import io
from jd  import JobDescriptionExtractor
from parser import ResumeParser

class AdvancedAIInterviewer:
    def __init__(self, openai_api_key,resume_path, job_url):
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=openai_api_key)


        self.resume_parser = ResumeParser(openai_api_key)
        self.job_extractor = JobDescriptionExtractor(openai_api_key)



        
        
        # Parse resume and job description
        try:
            self.resume_data = self.resume_parser.parse_resume(resume_path)
            print(self.resume_data)
            self.job_details = self.job_extractor.extract_job_details(job_url)
            print(self.job_details)
        except Exception as e:
            print(f"Error parsing resume or job description: {e}")
            self.resume_data = {}
            self.job_details = {}
        
        # Initialize Whisper model for better speech recognition
        self.whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        
        # Initialize WebRTC VAD for better voice activity detection
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (max)
        
        # Audio settings
        self.RATE = 16000
        self.CHUNK = 480  # 30ms at 16kHz
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Initialize pygame mixer for better audio playback
        pygame.mixer.init(44100, -16, 2, 2048)
        
        # Audio processing queues
        self.audio_queue = queue.Queue()
        self.speech_queue = queue.Queue()
        
        # State management
        self.is_ai_speaking = False
        self.is_listening = False
        self.silence_threshold = 0.03
        self.min_speech_duration = 0.5
        self.silence_duration = 1.0
        self.last_speech_time = datetime.now()
        self.conversation_history = []
        self.active_connections = set()

        
        self.interview_evaluation = None
        
        # Dynamic threshold adjustment
        self.energy_history = []
        self.adaptive_threshold = 0.1
        
        # Response caching for faster interactions
        self.response_cache = {}
        
    async def dynamic_vad_threshold(self, audio_data):
        """Dynamically adjust voice activity detection threshold"""
        energy = np.abs(np.frombuffer(audio_data, dtype=np.int16)).mean()
        self.energy_history.append(energy)
        
        if len(self.energy_history) > 50:
            self.energy_history.pop(0)
            self.adaptive_threshold = np.mean(self.energy_history) * 1.1
        
        return energy > self.adaptive_threshold

    async def process_audio_chunk(self, audio_chunk):
        """Process audio chunk with WebRTC VAD and energy detection"""
        is_speech = self.vad.is_speech(audio_chunk, self.RATE)
        energy_threshold_met = await self.dynamic_vad_threshold(audio_chunk)
        
        return is_speech and energy_threshold_met



    
    async def enhanced_speech_recognition(self, audio_data):
        """Improved speech recognition using Whisper"""
        try:
        # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                with wave.open(tmp_file.name, 'wb') as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    wf.writeframes(audio_data)
                
                # Use Whisper with more advanced transcription parameters
                segments, info = self.whisper_model.transcribe(
                    tmp_file.name,
                    beam_size=10,  # Increased beam size for better accuracy
                    best_of=5,     # Consider top 5 transcription candidates
                    word_timestamps=True,
                    condition_on_previous_text=True,  # Improve context understanding
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Multiple temperature values for robust transcription
                    compression_ratio_threshold=2.4,  # Stricter quality filtering
                    log_prob_threshold=-1.0
                )
                
                # Combine segments with better handling of punctuation and context
                transcribed_text = " ".join([
                    segment.text.strip() for segment in segments 
                    if segment.text.strip() and len(segment.text.strip()) > 2
                ])
                
            os.unlink(tmp_file.name)
            return transcribed_text
        
        except Exception as e:
            print(f"Enhanced Speech Recognition Error: {str(e)}")
            return ""

    

    async def advanced_audio_processing(self, websocket):
        """Enhanced audio processing with better noise handling and speech detection"""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        audio_buffer = []
        silence_frames = 0
        is_speaking = False
        
        while websocket in self.active_connections:
            if self.is_ai_speaking:
                await asyncio.sleep(0.1)
                continue
            
            try:
                audio_chunk = stream.read(self.CHUNK, exception_on_overflow=False)
                is_voice = await self.process_audio_chunk(audio_chunk)
                
                if is_voice:
                    if not is_speaking:
                        is_speaking = True
                        await websocket.send(json.dumps({"type": "listening_status", "status": True}))
                    
                    audio_buffer.append(audio_chunk)
                    silence_frames = 0
                else:
                    silence_frames += 1
                    
                    if is_speaking:
                        audio_buffer.append(audio_chunk)
                    
                    # Check if speech ended
                    if silence_frames > int(self.silence_duration * self.RATE / self.CHUNK):
                        if is_speaking and len(audio_buffer) > int(self.min_speech_duration * self.RATE / self.CHUNK):
                            # Process complete utterance
                            audio_data = b''.join(audio_buffer)
                            text = await self.enhanced_speech_recognition(audio_data)
                            
                            if text:
                                await self.handle_speech_input(text, websocket)
                        
                        # Reset state
                        audio_buffer = []
                        is_speaking = False
                        await websocket.send(json.dumps({"type": "listening_status", "status": False}))
                
            except Exception as e:
                print(f"Audio processing error: {str(e)}")
                await asyncio.sleep(0.1)
        
        stream.stop_stream()
        stream.close()

    async def generate_response(self, user_input):
        """Generate a response using OpenAI's GPT-40-mini"""
        try:
            messages = [
                {"role": "system", "content": self.get_enhanced_system_prompt()},
                *self.conversation_history[-5:],  # Keep last 5 exchanges for context
                {"role": "user", "content": user_input}
            ]
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use more advanced model for better responses
                messages=messages,
                temperature=0.7,
                max_tokens=150,  # Increased token limit for more comprehensive responses
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Response Generation Error: {str(e)}")
            return "I apologize, but I encountered an error generating a response."    

    async def handle_speech_input(self, text, websocket):
        """Handle speech input with improved response generation"""
        try:
            await websocket.send(json.dumps({
                "type": "speech",
                "text": text
            }))


            is_ending = await self.detect_interview_termination(text)
            
            # Generate and speak AI response
            response = await self.generate_response(text)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Send response chunks for smoother interaction
            chunk_size = 50
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i + chunk_size]
                await websocket.send(json.dumps({
                    "type": "ai_response",
                    "text": chunk
                }))
                await asyncio.sleep(0.05)
            
            await self.enhanced_text_to_speech(response, websocket)

            if is_ending:
                self.interview_evaluation = await self.post_interview_evaluation()
                await websocket.send(json.dumps({
                    "type": "interview_concluded",
                    "evaluation": self.interview_evaluation
                }))
            
        except Exception as e:
            print(f"Speech input handling error: {str(e)}")

    def get_enhanced_system_prompt(self):
        """Get enhanced system prompt with better interaction guidelines"""
        
        
    
        
        
        
        
        
        # Construct a dynamic, context-aware prompt
        personalized_prompt = f"""
        You are an expert technical  and behavioral all types of interviewer for conducting a interview for any role any company any candidate and be perfect immersive very personalized.

    Candidate Context:
    - Full Name: pick name from {self.resume_data}
    - All Skills: pick all skills from {self.resume_data}
    - Education: pick education  details from {self.resume_data}
    - Experience: pick experience details from {self.resume_data}
    - Projects: pick projects details from {self.resume_data}

    Company Context:
    - Company Name: {self.job_details}
    - Role: {self.job_details}
    - Key Responsibilities: pick responsibilities from {self.job_details}
    - Required Technical Skills: Take all the skills required from {self.job_details}   

    Job Requirements:
    Role: {self.job_details} at company name:- {self.job_details}
    Key Responsibilities:
      pick responsibilities from {self.job_details}

    Required Technical Skills:
    Take all the skills required from {self.job_details}

    Interview Objectives:
    1. Thoroughly assess technical competence and make different strategies  dynamically very personalized based on role and candidates  data from {self.resume_data} and {self.job_details} It should so immersive and simulate actual hiring of the company interviewing that candidate.
    2. Validate alignment with job requirements and company culture completely to recruit for that role for this candidate with {self.resume_data}
    3. Evaluate problem-solving and communication skills and all such needful whatever based on the role type ,candiate type and many more factors.
    4.YOU should no when to stop the interview also based on the candidate if he is too low to the job standarsds or anything such rules 
     you can stop further interview by concluding professionally.
    5. YOU should first think of ways how this canidate should be interviewed perfectly according to role and company based on {self.job_details}
    originally to recruit for that role and respnsilibilyt .
    6. You should make very very good question short and conscise according to role perfectly to recruit for that and 
    think of ways how you can assess this candidate for the role what questions and skills to be questioned and evaluated etc. 

    Interview Strategy:
    - Maintain a professional, conversational tone
    - Ask precise, role-specific technical questions
    - Use behavioral interview techniques
    - Assess depth of knowledge and practical experience
    - Provide constructive, contextual feedback
    -DO NOT GIVE MULTIPLE QUESTIONS AT A TIME ASK 1 or 2 QUESTIONS AT A TIMEnot more than 2 questions at a time maximum.
    -Go step by step if needed for mutiple questions after answering one go to next one but dont give them once.

     Example REMEMBER THIS IS JUST A SAMPLE  Interview Stages for reference:
    1. Warm Introduction
       - Confirm candidate's background
       - Build rapport
       - Explain interview process

    2. Technical Skill Assessment
       - Deep dive into technical skills
       - Practical coding/problem-solving scenarios
       - Verify claimed expertise

    3. Work Experience Exploration
       - Discuss past projects
       - Understand problem-solving approach
       - Evaluate real-world application of skills

    4. Behavioral Evaluation
       - Team collaboration scenarios
       - Handling challenges
       - Leadership and adaptability

    5. Career Alignment
       - Long-term career goals
       - Motivation for this role
       - Potential growth trajectory

    Interaction Guidelines:
    - Be precise and job-relevant
    - Avoid speculative or hypothetical scenarios
    - Focus on factual, demonstrable skills
    - Maintain professional boundaries
    - Provide clear, constructive feedback

    Strict Instructions:
    - Do NOT hallucinate or fabricate information
    - If uncertain about a detail, ask clarifying questions
    - Stay within the scope of the candidate's actual experience.
    -DO NOT HALLUCINATE OR FABRICATE OR ASSUME CANDIDATES RESPONSE AT ALL.
    -ALWAYS TRY TO be within the interview CONTEXT.DO NOT GO OUT OF CONTEXT.
    - Maintain ethical and professional interview standards

    Expected Interview Tone:
    - Professional yet conversational
    - Respectful and encouraging
    - Focused on mutual understanding

        """
        
        return personalized_prompt.strip()
    async def detect_interview_termination(self,user_input):

        termination_indicators = [
        "thank you",
        "that's all",
        "no more questions",
        "interview is over",
        "we're done",
        "finished",
        "complete",
        "end the interview"
    ]
    
        # Convert input to lowercase for case-insensitive matching
        normalized_input = user_input.lower()
        
        # Check if any termination indicators are present
        if any(indicator in normalized_input for indicator in termination_indicators):
            return True
        
        # Additional context-based termination detection using LLM
        try:
            termination_check_prompt = f"""
            Analyze the following conversation context and determine if the interview is likely ending:
            
            Recent Conversation:
            {self.conversation_history[-5:]}
            
            Latest User Input: {user_input}
            
            Is this conversation showing strong signals of interview conclusion?
            Respond with only 'YES' or 'NO'.
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert interview analysis AI."},
                    {"role": "user", "content": termination_check_prompt}
                ],
                max_tokens=5,
                temperature=0.2
            )
            
            llm_decision = response.choices[0].message.content.strip().upper()
            return llm_decision == 'YES'
        
        except Exception as e:
            print(f"Termination detection error: {e}")
            return False
    async def post_interview_evaluation(self):
        """Post-interview evaluation using GPT-4"""
        try:
            evaluation_prompt = f"""
            Perform a comprehensive post-interview evaluation based on the following context:

            Job Details: {self.job_details}
            Candidate Resume: {self.resume_data}
            Interview Transcript: {self.conversation_history}

            Provide a detailed, objective, and professional evaluation focusing on:

            1. Technical Competence Evaluation
            2. Soft Skills Assessment
            3. Cultural Fit
            4. Overall Interview Performance
            5. Hiring Recommendation

            Requirements:
            - Use data-driven insights from the interview
            - Compare candidate's skills with job requirements
            - Provide actionable feedback
            - Make a clear hiring recommendation
            - Be precise and professional

            
            Output Instructions:
            - Provide a clear, objective, and professional evaluation
            - Use data-driven insights from the interview
            - Compare candidate's skills with job requirements
            - Provide actionable feedback
            - Make a clear hiring recommendation
            - Be precise and professional
            - Focus on factual, demonstrable skills
            -Tell them where they are good at and where they could improve to crack this particular role with the company.
            -DO NOT HALLUCINATE OR FABRICATE OR ASSUME CANDIDATES RESPONSE AT ALL.
            -ALWAYS TRY TO be within the interview CONTEXT.DO NOT GO OUT OF CONTEXT.
            - Maintain ethical and professional interview standards

            -Output Format:
        STRICTLY GIVE IT IN JSON FORMAT, DO NOT INLCUDE any ``` backticks or anything else.
                         keep these fields inside json :
                         "technical_match"
                        "soft_skill_match" 
                        "performance_score"
                        "overall_performance"
                        "recommendation" 
                        "hiring_decision"

            """

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert technical interviewer and talent acquisition specialist with all capabilities."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.5
                
            )

            # Parse the JSON response
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation

        except Exception as e:
            print(f"Post-interview evaluation error: {str(e)}")
            return {
                "error": "Unable to generate evaluation",
                "details": str(e)
            }

    async def enhanced_text_to_speech(self, text, websocket):
        """Enhanced text-to-speech with better voice quality and timing"""
        try:
            self.is_ai_speaking = True
            await websocket.send(json.dumps({"type": "ai_speaking", "status": True}))
            
            # Use entire text in one TTS generation for maximum naturalness
            try:
                # 
                response = await self.client.audio.speech.create(
                    model="tts-1",  
                    voice="nova",      # Choose a natural-sounding voice
                    input=text,
                    speed=1.0           # Natural speaking speed
                )
                
                # Directly stream audio without file intermediary
                audio_task = asyncio.create_task(response.aread())
            
            # While audio is being generated, start sending text chunks
                # chunk_size = 50
                # for i in range(0, len(text), chunk_size):
                #     chunk = text[i:i + chunk_size]
                #     await websocket.send(json.dumps({
                #         "type": "ai_response",
                #         "text": chunk
                #     }))
                      # Small delay to simulate typing
                
                # Wait for audio to be fully generated
                audio_stream = await audio_task
                
                # Use pygame for immediate playback
                pygame.mixer.init()
                audio_file = io.BytesIO(audio_stream)
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # Wait for entire audio to complete
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
            
            except Exception as audio_error:
                print(f"Audio Playback Error: {audio_error}")
        
            self.is_ai_speaking = False
            await websocket.send(json.dumps({"type": "ai_speaking", "status": False}))
    
        except Exception as e:
            print(f"TTS Enhancement Error: {str(e)}")
            self.is_ai_speaking = False

    async def handle_websocket(self, websocket):
        """Enhanced WebSocket handler with better connection management"""
        self.active_connections.add(websocket)
        
        welcome_msg = "Hello! I'm your AI interviewer today. Please start with your introduction."
        await websocket.send(json.dumps({"type": "ai_response", "text": welcome_msg}))
        await self.enhanced_text_to_speech(welcome_msg, websocket)
        
        try:
            await self.advanced_audio_processing(websocket)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.active_connections.remove(websocket)

    async def start_server(self):
        """Start WebSocket server with enhanced error handling"""
        try:
            server = await websockets.serve(
                self.handle_websocket,
                "localhost",
                8765,
                ping_interval=30,
                ping_timeout=10
            )
            
            print("Enhanced AI Interviewer ready on ws://localhost:8765")
            await server.wait_closed()
            
        except Exception as e:
            print(f"Server error: {str(e)}")

def main():
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    resume_path = ""  # Replace with actual resume path
    job_url = ""  # Replace with actual job URL
    
    if not openai_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    interviewer = AdvancedAIInterviewer(openai_key, resume_path, job_url)
    
    try:
        asyncio.run(interviewer.start_server())
    except KeyboardInterrupt:
        print("\nShutting down Enhanced AI Interviewer...")

if __name__ == "__main__":
    main()