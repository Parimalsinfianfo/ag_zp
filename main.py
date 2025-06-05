from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List, Dict, Any
import sqlite3
import json
import re
import time
import hashlib
import smtplib
import os
import uuid
import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio
from contextlib import asynccontextmanager

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.chains.question_answering import load_qa_chain
from bs4 import BeautifulSoup

# Environment Configuration
# Replace hardcoded values with environment variables
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Configuration
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
EMAIL_CONFIG = {
    "provider": "gmail",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "email": os.getenv("EMAIL_USER"),  # Use env vars
    "password": os.getenv("EMAIL_PASSWORD"),  # Use env vars
    "enabled": True
}

# Domain-specific URLs
URLS = [
    "https://www.zedprodigital.com/",
    "https://www.zedprodigital.com/our-ai-services/",
    "https://www.zedprodigital.com/projects-case-studies/",
    "https://www.zedprodigital.com/odoo-partners/",
    "https://www.zedprodigital.com/lets-get-connected/"
]
app.mount("/static", StaticFiles(directory=os.path.join(os.getcwd(), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.getcwd(), "templates"))
# Global variables for AI components
llm = None
embeddings = None
vectorstore = None
qa_chain = None
memory_store = {}

# Pydantic Models
class VisitorInfo(BaseModel):
    name: str
    email: EmailStr
    phone: str
    city: str
    session_id: Optional[str] = None
    interest_domain: Optional[str] = None
    
    @validator('phone')
    def validate_phone(cls, v):
        cleaned_phone = re.sub(r'[^\d+]', '', v)
        if not re.match(r'^[\+]?[1-9][\d]{3,14}$', cleaned_phone) or len(cleaned_phone) < 10:
            raise ValueError('Invalid phone number format')
        return v

class ChatMessage(BaseModel):
    message: str
    session_id: str

class EmailCampaign(BaseModel):
    subject: str
    content: str
    recipients: List[str]
    campaign_type: str = "custom"

class KnowledgeBase(BaseModel):
    category: str
    question: str
    answer: str
    keywords: str

# Database Setup
def init_database():
    db_path = os.path.join(os.getcwd(), 'ai_service_agent.db')  # Absolute path
    conn = sqlite3.connect(db_path)
    # ... rest of the code
    
    tables = [
        '''CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            name TEXT, email TEXT, phone TEXT, city TEXT,
            visit_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pages_visited TEXT, time_spent INTEGER DEFAULT 0,
            status TEXT DEFAULT 'new', lead_score INTEGER DEFAULT 0,
            ip_address TEXT, user_agent TEXT,
            interest_domain TEXT
        )''',
        '''CREATE TABLE IF NOT EXISTS chat_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, message TEXT, response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_type TEXT DEFAULT 'user'
        )''',
        '''CREATE TABLE IF NOT EXISTS company_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT, question TEXT, answer TEXT, keywords TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''',
        '''CREATE TABLE IF NOT EXISTS email_campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visitor_id INTEGER, email_type TEXT, subject TEXT, content TEXT,
            sent_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, status TEXT DEFAULT 'sent',
            recipient_email TEXT
        )''',
        '''CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE, visitors_count INTEGER DEFAULT 0,
            leads_generated INTEGER DEFAULT 0, emails_sent INTEGER DEFAULT 0,
            conversions INTEGER DEFAULT 0, page_views INTEGER DEFAULT 0
        )''',
        '''CREATE TABLE IF NOT EXISTS page_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, page_url TEXT, time_spent INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )'''
    ]
    
    for table in tables:
        cursor.execute(table)
    
    # Insert sample company data
    cursor.execute("SELECT COUNT(*) FROM company_data")
    if cursor.fetchone()[0] == 0:
        sample_data = [
            ("services", "What services do you offer?", "We offer AI solutions, web development, mobile app development, digital marketing, cloud services, and automation solutions.", "services, offerings, solutions"),
            ("pricing", "What are your pricing plans?", "Our pricing varies based on project scope. We offer competitive rates starting from $500 for basic solutions to $50,000+ for enterprise AI implementations.", "pricing, cost, rates"),
            ("contact", "How can I contact you?", "You can reach us via email at contact@zedprodigital.com, call us at +91-9876543210, or schedule a meeting through our website.", "contact, reach, support"),
            ("experience", "How experienced are you?", "We have over 8 years of experience in AI and digital solutions with a team of 25+ professionals serving 200+ clients globally.", "experience, years, expertise"),
            ("location", "Where are you located?", "We have offices in Mumbai, Pune, and Bangalore, with remote teams across India and international presence.", "location, office, address"),
            ("ai_services", "What AI services do you provide?", "We provide custom AI chatbots, machine learning solutions, natural language processing, computer vision, predictive analytics, and AI automation.", "AI, artificial intelligence, machine learning"),
            ("support", "Do you provide support?", "Yes, we provide 24/7 technical support, maintenance services, and dedicated account management for all our clients.", "support, maintenance, help"),
            ("industries", "Which industries do you serve?", "We serve healthcare, finance, e-commerce, education, manufacturing, real estate, and technology sectors with tailored AI solutions.", "industries, sectors, domains")
        ]
        cursor.executemany("INSERT INTO company_data (category, question, answer, keywords) VALUES (?, ?, ?, ?)", sample_data)
        conn.commit()
    
    conn.close()

# Web Scraping Functions
async def scrape_website_content(url: str) -> str:
    """Scrape and extract main content from a webpage"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unnecessary elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.decompose()
        
        # Extract main content
        content = ""
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            content = main_content.get_text(separator='\n', strip=True)
        else:
            content = soup.get_text(separator='\n', strip=True)
        
        # Clean up text
        content = re.sub(r'\n\s*\n', '\n\n', content)
        return content[:10000]  # Limit to first 10,000 characters
    
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

async def scrape_domain_content(urls: List[str]) -> List[Document]:
    """Scrape multiple URLs and return as documents"""
    documents = []
    for url in urls:
        print(f"Scraping: {url}")
        content = await scrape_website_content(url)
        if content:
            metadata = {"source": url}
            documents.append(Document(page_content=content, metadata=metadata))
    return documents

# Initialize AI Components
async def initialize_ai_components():
    global llm, embeddings, vectorstore, qa_chain
    
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Skipping AI initialization.")
        return
    
    try:
        # ... existing initialization code
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192",
            temperature=0.7,
            max_tokens=500
        )
        
        # Initialize Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create knowledge base
        await create_knowledge_base()
        
        print("AI components initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing AI components: {e}")

async def create_knowledge_base():
    global vectorstore, qa_chain, embeddings, llm
    
    try:
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer, category, keywords FROM company_data")
        knowledge_data = cursor.fetchall()
        conn.close()
        
        # Scrape domain content
        domain_docs = await scrape_domain_content(URLS)
        
        documents = []
        for question, answer, category, keywords in knowledge_data:
            doc_content = f"Category: {category}\nQuestion: {question}\nAnswer: {answer}\nKeywords: {keywords}"
            documents.append(Document(page_content=doc_content, metadata={"category": category}))
        
        # Add scraped domain content
        documents.extend(domain_docs)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        
        if embeddings and splits:
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            # Create QA Chain
            prompt_template = """You are ZedPro Digital's professional AI assistant. You help visitors learn about our AI and digital services.
            Use the following context to answer questions accurately and helpfully about ZedPro Digital.
            
            Context: {context}
            Chat History: {chat_history}
            Human: {question}
            
            Guidelines:
            - Be friendly, professional, and helpful
            - Use the context to provide accurate information about ZedPro Digital's services
            - If you don't know something specific, offer to connect them with our team
            - Ask relevant follow-up questions to understand their project needs
            - Keep responses concise but informative
            - Focus on AI solutions, web development, and digital transformation services
            - Encourage them to share their contact information for personalized assistance
            
            Assistant: """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )
            
            # Create memory for each session
            memory = ConversationBufferWindowMemory(
                k=10,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=True
            )
            
            print("Knowledge base created successfully!")
        
    except Exception as e:
        print(f"Error creating knowledge base: {e}")

# Utility Functions
def validate_email(email: str) -> bool:
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None

def validate_phone(phone: str) -> bool:
    cleaned_phone = re.sub(r'[^\d+]', '', phone)
    return re.match(r'^[\+]?[1-9][\d]{3,14}$', cleaned_phone) and len(cleaned_phone) >= 10

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

async def send_email(to_email: str, subject: str, body: str, email_type: str = "greeting"):
    if not EMAIL_CONFIG.get('enabled', False):
        print("Email sending is disabled")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['email']
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['email'], to_email, msg.as_string())
        server.quit()
        
        # Log email
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO email_campaigns (visitor_id, email_type, subject, content, status, recipient_email)
                         VALUES (?, ?, ?, ?, ?, ?)''', 
                         (0, email_type, subject, body, "sent", to_email))
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False

async def get_ai_response(message: str, session_id: str) -> str:
    global qa_chain, memory_store
    
    if not qa_chain:
        return "I'm here to help you learn about ZedPro Digital's AI and digital services! How can I assist you today?"
    
    try:
        # Get or create memory for this session
        if session_id not in memory_store:
            memory_store[session_id] = ConversationBufferWindowMemory(
                k=10,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
        result = qa_chain.invoke({"question": message})
        response = result.get('answer', 'I apologize, but I could not process your request. Please try rephrasing your question.')
        
        # Save to database
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO chat_conversations (session_id, message, response, message_type)
                         VALUES (?, ?, ?, ?)''', 
                         (session_id, message, response, "user"))
        conn.commit()
        conn.close()
        
        return response
        
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "I'm here to help! Could you please rephrase your question about our AI and digital services?"

async def update_analytics():
    conn = sqlite3.connect('ai_service_agent.db')
    cursor = conn.cursor()
    today = datetime.now().date()
    
    # Get today's stats
    visitors_today = cursor.execute("SELECT COUNT(*) FROM visitors WHERE DATE(visit_time) = ?", (today,)).fetchone()[0]
    leads_today = cursor.execute("SELECT COUNT(*) FROM visitors WHERE DATE(visit_time) = ? AND email IS NOT NULL", (today,)).fetchone()[0]
    emails_today = cursor.execute("SELECT COUNT(*) FROM email_campaigns WHERE DATE(sent_time) = ?", (today,)).fetchone()[0]
    page_views_today = cursor.execute("SELECT COUNT(*) FROM page_tracking WHERE DATE(timestamp) = ?", (today,)).fetchone()[0]
    
    cursor.execute('''INSERT OR REPLACE INTO analytics (date, visitors_count, leads_generated, emails_sent, page_views)
                     VALUES (?, ?, ?, ?, ?)''', 
                     (today, visitors_today, leads_today, emails_today, page_views_today))
    conn.commit()
    conn.close()

# Startup and Shutdown Events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting AI Service Agent...")
    init_database()
    await initialize_ai_components()
    print("AI Service Agent started successfully!")
    yield
    # Shutdown
    print("Shutting down AI Service Agent...")

# FastAPI App
app = FastAPI(
    title="AI Service Agent - ZedPro Digital",
    description="AI-powered lead generation and customer engagement system",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/visitor/register")
async def register_visitor(visitor: VisitorInfo, background_tasks: BackgroundTasks, request: Request):
    try:
        session_id = visitor.session_id or str(uuid.uuid4())
        ip_address = get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        
        cursor.execute('''INSERT OR REPLACE INTO visitors 
                         (session_id, name, email, phone, city, visit_time, ip_address, user_agent, interest_domain)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                         (session_id, visitor.name, visitor.email, visitor.phone, 
                          visitor.city, datetime.now(), ip_address, user_agent, visitor.interest_domain))
        conn.commit()
        conn.close()
        
        # Send welcome email in background
        welcome_subject = f"Welcome to ZedPro Digital, {visitor.name}!"
        welcome_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #667eea;">Welcome to ZedPro Digital, {visitor.name}!</h2>
                <p>Thank you for your interest in our AI and digital solutions!</p>
                <p>We specialize in:</p>
                <ul>
                    <li>🤖 Custom AI Solutions & Chatbots</li>
                    <li>🌐 Web & Mobile Development</li>
                    <li>📊 Data Analytics & Machine Learning</li>
                    <li>☁️ Cloud Services & Automation</li>
                    <li>📱 Digital Marketing Solutions</li>
                </ul>
                <p>Our team will reach out to you shortly to discuss your project requirements.</p>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p><strong>Quick Connect:</strong></p>
                    <p>📧 Email: contact@zedprodigital.com</p>
                    <p>📞 Phone: +91-9876543210</p>
                    <p>🌐 Website: https://www.zedprodigital.com</p>
                </div>
                <p>Best regards,<br><strong>ZedPro Digital Team</strong></p>
            </div>
        </body>
        </html>
        """
        
        background_tasks.add_task(send_email, visitor.email, welcome_subject, welcome_body, "welcome")
        background_tasks.add_task(update_analytics)
        
        return {"success": True, "message": "Visitor registered successfully", "session_id": session_id}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chat")
async def chat_with_ai(chat_data: ChatMessage):
    try:
        response = await get_ai_response(chat_data.message, chat_data.session_id)
        return {"success": True, "response": response, "session_id": chat_data.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/api/track/page")
async def track_page_visit(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id", str(uuid.uuid4()))
        page_url = data.get("page_url", "")
        time_spent = data.get("time_spent", 0)
        
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO page_tracking (session_id, page_url, time_spent)
                         VALUES (?, ?, ?)''', (session_id, page_url, time_spent))
        conn.commit()
        conn.close()
        
        return {"success": True, "message": "Page visit tracked"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics():
    try:
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        # Today's stats
        visitors_today = cursor.execute("SELECT COUNT(*) FROM visitors WHERE DATE(visit_time) = ?", (today,)).fetchone()[0]
        leads_today = cursor.execute("SELECT COUNT(*) FROM visitors WHERE DATE(visit_time) = ? AND email IS NOT NULL", (today,)).fetchone()[0]
        
        # This week's stats
        visitors_week = cursor.execute("SELECT COUNT(*) FROM visitors WHERE DATE(visit_time) >= ?", (week_ago,)).fetchone()[0]
        leads_week = cursor.execute("SELECT COUNT(*) FROM visitors WHERE DATE(visit_time) >= ? AND email IS NOT NULL", (week_ago,)).fetchone()[0]
        
        # This month's stats
        visitors_month = cursor.execute("SELECT COUNT(*) FROM visitors WHERE DATE(visit_time) >= ?", (month_ago,)).fetchone()[0]
        leads_month = cursor.execute("SELECT COUNT(*) FROM visitors WHERE DATE(visit_time) >= ? AND email IS NOT NULL", (month_ago,)).fetchone()[0]
        
        # Total stats
        total_visitors = cursor.execute("SELECT COUNT(*) FROM visitors").fetchone()[0]
        total_leads = cursor.execute("SELECT COUNT(*) FROM visitors WHERE email IS NOT NULL").fetchone()[0]
        total_emails = cursor.execute("SELECT COUNT(*) FROM email_campaigns").fetchone()[0]
        
        # Recent visitors
        recent_visitors = cursor.execute("""
            SELECT name, email, city, visit_time, status, interest_domain 
            FROM visitors 
            WHERE visit_time >= datetime('now', '-7 days') 
            ORDER BY visit_time DESC LIMIT 10
        """).fetchall()
        
        # Daily visitors for the last 30 days
        daily_visitors = cursor.execute("""
            SELECT DATE(visit_time) as date, COUNT(*) as visitors,
                   COUNT(CASE WHEN email IS NOT NULL THEN 1 END) as leads
            FROM visitors 
            WHERE DATE(visit_time) >= ? 
            GROUP BY DATE(visit_time) 
            ORDER BY date
        """, (month_ago,)).fetchall()
        
        conn.close()
        
        return {
            "success": True,
            "data": {
                "today": {"visitors": visitors_today, "leads": leads_today},
                "week": {"visitors": visitors_week, "leads": leads_week},
                "month": {"visitors": visitors_month, "leads": leads_month},
                "total": {"visitors": total_visitors, "leads": total_leads, "emails": total_emails},
                "recent_visitors": [
                    {"name": v[0], "email": v[1], "city": v[2], "visit_time": v[3], "status": v[4], "interest_domain": v[5]}
                    for v in recent_visitors
                ],
                "daily_chart": [
                    {"date": d[0], "visitors": d[1], "leads": d[2]}
                    for d in daily_visitors
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/leads")
async def get_leads(status: Optional[str] = None, city: Optional[str] = None):
    try:
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        
        query = "SELECT * FROM visitors WHERE email IS NOT NULL"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if city:
            query += " AND city = ?"
            params.append(city)
        
        query += " ORDER BY visit_time DESC"
        
        leads = cursor.execute(query, params).fetchall()
        conn.close()
        
        leads_data = []
        for lead in leads:
            leads_data.append({
                "id": lead[0],
                "session_id": lead[1],
                "name": lead[2],
                "email": lead[3],
                "phone": lead[4],
                "city": lead[5],
                "visit_time": lead[6],
                "status": lead[9],
                "lead_score": lead[10],
                "interest_domain": lead[12]
            })
        
        return {"success": True, "leads": leads_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/leads/{lead_id}/status")
async def update_lead_status(lead_id: int, request: Request):
    try:
        data = await request.json()
        new_status = data.get("status")
        
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        cursor.execute("UPDATE visitors SET status = ? WHERE id = ?", (new_status, lead_id))
        conn.commit()
        conn.close()
        
        return {"success": True, "message": "Lead status updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/email/campaign")
async def send_email_campaign(campaign: EmailCampaign, background_tasks: BackgroundTasks):
    try:
        success_count = 0
        
        for recipient_email in campaign.recipients:
            personalized_content = campaign.content.replace("{email}", recipient_email)
            success = await send_email(recipient_email, campaign.subject, personalized_content, campaign.campaign_type)
            if success:
                success_count += 1
        
        return {
            "success": True, 
            "message": f"Campaign sent to {success_count}/{len(campaign.recipients)} recipients"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/add")
async def add_knowledge(knowledge: KnowledgeBase, background_tasks: BackgroundTasks):
    try:
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO company_data (category, question, answer, keywords)
                         VALUES (?, ?, ?, ?)''', 
                         (knowledge.category, knowledge.question, knowledge.answer, knowledge.keywords))
        conn.commit()
        conn.close()
        
        # Rebuild knowledge base in background
        background_tasks.add_task(create_knowledge_base)
        
        return {"success": True, "message": "Knowledge added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/knowledge")
async def get_knowledge():
    try:
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        knowledge = cursor.execute("SELECT * FROM company_data ORDER BY category, id").fetchall()
        conn.close()
        
        knowledge_data = []
        for k in knowledge:
            knowledge_data.append({
                "id": k[0],
                "category": k[1],
                "question": k[2],
                "answer": k[3],
                "keywords": k[4],
                "created_at": k[5]
            })
        
        return {"success": True, "knowledge": knowledge_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    try:
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        history = cursor.execute("""
            SELECT message, response, timestamp, message_type 
            FROM chat_conversations 
            WHERE session_id = ? 
            ORDER BY timestamp
        """, (session_id,)).fetchall()
        conn.close()
        
        chat_data = []
        for h in history:
            chat_data.append({
                "message": h[0],
                "response": h[1],
                "timestamp": h[2],
                "type": h[3]
            })
        
        return {"success": True, "history": chat_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_components": {
            "llm": llm is not None,
            "embeddings": embeddings is not None,
            "vectorstore": vectorstore is not None,
            "qa_chain": qa_chain is not None
        }
    }
@app.on_event("shutdown")
async def shutdown_event():
    print("Waiting for background tasks to complete...")
    await asyncio.sleep(2)  # Grace period

# Render deployment entry point
# ... rest of your code ...

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render compatibility
    uvicorn.run(app, host="0.0.0.0", port=port)
