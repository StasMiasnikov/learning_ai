# Programming Essentials for LLM and MCP Development

This section covers the essential programming skills, tools, and frameworks needed for building Large Language Model agents and Multi-Agent Collaboration Platforms.

## 🐍 Advanced Python Programming

While basic Python knowledge is assumed, working with LLMs and multi-agent systems requires advanced Python concepts.

### Object-Oriented Programming for AI Systems

**Classes and Inheritance**:
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.memory = {}
        self.is_active = False
    
    @abstractmethod
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """Process incoming message and return response"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return current agent status"""
        pass
    
    def add_memory(self, key: str, value: Any) -> None:
        """Store information in agent memory"""
        self.memory[key] = value
    
    def recall_memory(self, key: str) -> Optional[Any]:
        """Retrieve information from memory"""
        return self.memory.get(key)

class LLMAgent(BaseAgent):
    """LLM-powered agent implementation"""
    
    def __init__(self, name: str, model_name: str, system_prompt: str):
        super().__init__(name, ["text_generation", "reasoning", "analysis"])
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.conversation_history = []
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """Process message using LLM"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Simulate LLM call (replace with actual API call)
        response = await self._generate_response(message, context)
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate response using LLM (placeholder)"""
        # This would integrate with OpenAI, Hugging Face, or other LLM APIs
        await asyncio.sleep(0.1)  # Simulate API call delay
        return f"Response to: {message}"
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model_name,
            "is_active": self.is_active,
            "conversation_length": len(self.conversation_history),
            "memory_size": len(self.memory)
        }

# Usage example
async def main():
    agent = LLMAgent("Assistant", "gpt-4", "You are a helpful assistant")
    response = await agent.process_message("Hello!", {})
    print(f"Agent response: {response}")
    print(f"Agent status: {agent.get_status()}")

# asyncio.run(main())
```

### Advanced Python Features

**Decorators for Agent Functionality**:
```python
import functools
import time
from typing import Callable, Any
import logging

def log_agent_action(func: Callable) -> Callable:
    """Decorator to log agent actions"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        logging.info(f"Agent {self.name} starting action: {func.__name__}")
        
        try:
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            logging.info(f"Agent {self.name} completed {func.__name__} in {duration:.2f}s")
            return result
        except Exception as e:
            logging.error(f"Agent {self.name} failed {func.__name__}: {e}")
            raise
    
    return wrapper

def rate_limit(max_calls: int, time_window: int):
    """Decorator to rate limit agent actions"""
    def decorator(func: Callable) -> Callable:
        calls = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls outside time window
            calls[:] = [call_time for call_time in calls if now - call_time < time_window]
            
            if len(calls) >= max_calls:
                raise Exception(f"Rate limit exceeded: {max_calls} calls per {time_window}s")
            
            calls.append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Usage
class RateLimitedAgent(LLMAgent):
    @log_agent_action
    @rate_limit(max_calls=10, time_window=60)
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        return await super().process_message(message, context)
```

**Context Managers for Resource Management**:
```python
from contextlib import contextmanager, asynccontextmanager
import aiohttp

@asynccontextmanager
async def llm_api_client(api_key: str, base_url: str):
    """Context manager for LLM API client"""
    async with aiohttp.ClientSession(
        headers={"Authorization": f"Bearer {api_key}"}
    ) as session:
        try:
            yield session
        finally:
            # Cleanup code here
            pass

class EnhancedLLMAgent(BaseAgent):
    def __init__(self, name: str, api_key: str, base_url: str):
        super().__init__(name, ["text_generation"])
        self.api_key = api_key
        self.base_url = base_url
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        async with llm_api_client(self.api_key, self.base_url) as client:
            # Use client to make API calls
            response = await self._call_llm_api(client, message)
            return response
    
    async def _call_llm_api(self, client: aiohttp.ClientSession, message: str) -> str:
        # Implement actual API call
        return "API response"
```

## 🔄 Asynchronous Programming

Working with multiple agents requires understanding asynchronous programming for concurrent operations.

### asyncio Fundamentals

**Basic Async Operations**:
```python
import asyncio
import aiohttp
from typing import List, Coroutine

async def fetch_data(url: str) -> str:
    """Fetch data from URL asynchronously"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def process_multiple_requests(urls: List[str]) -> List[str]:
    """Process multiple requests concurrently"""
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# Multi-agent coordination
async def coordinate_agents(agents: List[BaseAgent], task: str) -> List[str]:
    """Coordinate multiple agents to work on a task"""
    tasks = [agent.process_message(task, {}) for agent in agents]
    responses = await asyncio.gather(*tasks)
    return responses
```

**Advanced Async Patterns**:
```python
import asyncio
from asyncio import Queue
from typing import AsyncGenerator

class MessageBroker:
    """Async message broker for agent communication"""
    
    def __init__(self):
        self.subscribers = {}
        self._running = False
    
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
    
    async def publish(self, topic: str, message: Any):
        """Publish message to topic"""
        if topic in self.subscribers:
            tasks = [callback(message) for callback in self.subscribers[topic]]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def start(self):
        """Start the message broker"""
        self._running = True
        while self._running:
            await asyncio.sleep(0.1)  # Keep broker alive
    
    def stop(self):
        """Stop the message broker"""
        self._running = False

# Producer-Consumer pattern for agent tasks
async def task_producer(queue: Queue, tasks: List[str]):
    """Produce tasks for agents to consume"""
    for task in tasks:
        await queue.put(task)
    
    # Signal completion
    await queue.put(None)

async def agent_consumer(queue: Queue, agent: BaseAgent) -> List[str]:
    """Agent consumes and processes tasks"""
    results = []
    
    while True:
        task = await queue.get()
        if task is None:  # Completion signal
            break
        
        result = await agent.process_message(task, {})
        results.append(result)
        queue.task_done()
    
    return results
```

## 🌐 Web Development for Agent Interfaces

Building web interfaces for multi-agent systems requires modern web development skills.

### FastAPI for Agent APIs

**RESTful API Design**:
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uuid

app = FastAPI(title="Multi-Agent System API", version="1.0.0")

# CORS middleware for web interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AgentMessage(BaseModel):
    content: str = Field(..., description="Message content")
    context: Dict[str, Any] = Field(default_factory=dict)
    agent_id: Optional[str] = Field(None, description="Target agent ID")

class AgentResponse(BaseModel):
    agent_id: str
    response: str
    timestamp: str
    status: str

class TaskRequest(BaseModel):
    task: str
    agents: List[str] = Field(..., description="Agent IDs to assign task to")
    parallel: bool = Field(True, description="Execute in parallel")

# Agent registry
agents: Dict[str, BaseAgent] = {}

@app.post("/agents/{agent_id}/message", response_model=AgentResponse)
async def send_message_to_agent(agent_id: str, message: AgentMessage):
    """Send message to specific agent"""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    try:
        response = await agent.process_message(message.content, message.context)
        return AgentResponse(
            agent_id=agent_id,
            response=response,
            timestamp=str(time.time()),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks/execute")
async def execute_task(task_request: TaskRequest, background_tasks: BackgroundTasks):
    """Execute task across multiple agents"""
    target_agents = [agents[aid] for aid in task_request.agents if aid in agents]
    
    if not target_agents:
        raise HTTPException(status_code=404, detail="No valid agents found")
    
    task_id = str(uuid.uuid4())
    
    if task_request.parallel:
        # Execute in parallel
        background_tasks.add_task(
            execute_parallel_task,
            task_id,
            target_agents,
            task_request.task
        )
    else:
        # Execute sequentially
        background_tasks.add_task(
            execute_sequential_task,
            task_id,
            target_agents,
            task_request.task
        )
    
    return {"task_id": task_id, "status": "started"}

@app.get("/agents")
async def list_agents():
    """List all available agents"""
    return {
        agent_id: agent.get_status()
        for agent_id, agent in agents.items()
    }

@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get specific agent status"""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return agents[agent_id].get_status()

# Background task functions
async def execute_parallel_task(task_id: str, agents: List[BaseAgent], task: str):
    """Execute task in parallel across agents"""
    try:
        results = await coordinate_agents(agents, task)
        # Store results (in production, use database)
        print(f"Task {task_id} completed: {len(results)} responses")
    except Exception as e:
        print(f"Task {task_id} failed: {e}")

async def execute_sequential_task(task_id: str, agents: List[BaseAgent], task: str):
    """Execute task sequentially across agents"""
    try:
        results = []
        for agent in agents:
            result = await agent.process_message(task, {})
            results.append(result)
        print(f"Task {task_id} completed sequentially: {len(results)} responses")
    except Exception as e:
        print(f"Task {task_id} failed: {e}")

# Startup event to initialize agents
@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    # Create sample agents
    agent1 = LLMAgent("agent_1", "gpt-4", "You are Agent 1")
    agent2 = LLMAgent("agent_2", "gpt-4", "You are Agent 2")
    
    agents["agent_1"] = agent1
    agents["agent_2"] = agent2
    
    print("Multi-Agent System API started")
```

### WebSocket for Real-time Communication

**Real-time Agent Communication**:
```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import json

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.websocket("/ws/agents/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time agent communication"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message
            if message_data["type"] == "agent_message":
                agent_id = message_data["agent_id"]
                content = message_data["content"]
                
                if agent_id in agents:
                    response = await agents[agent_id].process_message(content, {})
                    
                    # Send response back
                    response_data = {
                        "type": "agent_response",
                        "agent_id": agent_id,
                        "response": response,
                        "client_id": client_id
                    }
                    
                    await manager.send_personal_message(
                        json.dumps(response_data),
                        websocket
                    )
            
            elif message_data["type"] == "broadcast":
                # Broadcast to all connected clients
                await manager.broadcast(
                    json.dumps({
                        "type": "broadcast",
                        "message": message_data["content"],
                        "from": client_id
                    })
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

## 🗃️ Database Integration

Persistent storage is crucial for agent memory and system state.

### SQLAlchemy for Relational Data

**Database Models**:
```python
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime

Base = declarative_base()

class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True)
    name = Column(String(100), nullable=False)
    model_name = Column(String(100))
    system_prompt = Column(Text)
    capabilities = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="agent")
    memories = relationship("AgentMemory", back_populates="agent")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String, ForeignKey("agents.id"))
    user_message = Column(Text)
    agent_response = Column(Text)
    context = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    agent = relationship("Agent", back_populates="conversations")

class AgentMemory(Base):
    __tablename__ = "agent_memories"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String, ForeignKey("agents.id"))
    memory_key = Column(String(200))
    memory_value = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Relationships
    agent = relationship("Agent", back_populates="memories")

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True)
    description = Column(Text)
    assigned_agents = Column(JSON)
    status = Column(String(50), default="pending")
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

# Database connection
DATABASE_URL = "postgresql://username:password@localhost/agent_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
Base.metadata.create_all(bind=engine)
```

**Database Operations**:
```python
from sqlalchemy.orm import Session
from typing import List, Optional

class AgentRepository:
    """Repository pattern for agent data operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_agent(self, agent_id: str, name: str, model_name: str, 
                    system_prompt: str, capabilities: List[str]) -> Agent:
        """Create new agent in database"""
        db_agent = Agent(
            id=agent_id,
            name=name,
            model_name=model_name,
            system_prompt=system_prompt,
            capabilities=capabilities
        )
        self.db.add(db_agent)
        self.db.commit()
        self.db.refresh(db_agent)
        return db_agent
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        return self.db.query(Agent).filter(Agent.id == agent_id).first()
    
    def list_active_agents(self) -> List[Agent]:
        """List all active agents"""
        return self.db.query(Agent).filter(Agent.is_active == 1).all()
    
    def save_conversation(self, agent_id: str, user_message: str, 
                         agent_response: str, context: dict) -> Conversation:
        """Save conversation to database"""
        conversation = Conversation(
            agent_id=agent_id,
            user_message=user_message,
            agent_response=agent_response,
            context=context
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation
    
    def get_conversation_history(self, agent_id: str, limit: int = 100) -> List[Conversation]:
        """Get conversation history for agent"""
        return (self.db.query(Conversation)
                .filter(Conversation.agent_id == agent_id)
                .order_by(Conversation.timestamp.desc())
                .limit(limit)
                .all())
    
    def save_memory(self, agent_id: str, key: str, value: dict, 
                   expires_at: Optional[datetime] = None) -> AgentMemory:
        """Save agent memory"""
        memory = AgentMemory(
            agent_id=agent_id,
            memory_key=key,
            memory_value=value,
            expires_at=expires_at
        )
        self.db.add(memory)
        self.db.commit()
        self.db.refresh(memory)
        return memory
    
    def get_memory(self, agent_id: str, key: str) -> Optional[AgentMemory]:
        """Get agent memory by key"""
        return (self.db.query(AgentMemory)
                .filter(AgentMemory.agent_id == agent_id)
                .filter(AgentMemory.memory_key == key)
                .filter(
                    (AgentMemory.expires_at.is_(None)) |
                    (AgentMemory.expires_at > datetime.utcnow())
                )
                .first())

# Enhanced agent with database integration
class DatabaseAgent(BaseAgent):
    """Agent with database persistence"""
    
    def __init__(self, agent_id: str, name: str, model_name: str, 
                 system_prompt: str, db: Session):
        super().__init__(name, ["text_generation", "memory_persistence"])
        self.agent_id = agent_id
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.repo = AgentRepository(db)
        self.db = db
        
        # Load or create agent in database
        db_agent = self.repo.get_agent(agent_id)
        if not db_agent:
            self.repo.create_agent(agent_id, name, model_name, system_prompt, self.capabilities)
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """Process message with database persistence"""
        # Generate response (placeholder)
        response = f"Response to: {message}"
        
        # Save conversation to database
        self.repo.save_conversation(self.agent_id, message, response, context)
        
        return response
    
    def add_memory(self, key: str, value: Any, expires_at: Optional[datetime] = None) -> None:
        """Add memory with database persistence"""
        super().add_memory(key, value)
        self.repo.save_memory(self.agent_id, key, {"value": value}, expires_at)
    
    def recall_memory(self, key: str) -> Optional[Any]:
        """Recall memory from database"""
        db_memory = self.repo.get_memory(self.agent_id, key)
        if db_memory:
            return db_memory.memory_value.get("value")
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get status including database info"""
        conversation_count = (self.db.query(Conversation)
                             .filter(Conversation.agent_id == self.agent_id)
                             .count())
        memory_count = (self.db.query(AgentMemory)
                       .filter(AgentMemory.agent_id == self.agent_id)
                       .count())
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "model": self.model_name,
            "is_active": self.is_active,
            "conversation_count": conversation_count,
            "memory_count": memory_count
        }
```

### NoSQL with MongoDB

**Document-based Storage**:
```python
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from datetime import datetime
from typing import Dict, List, Optional
import json

class MongoDBManager:
    """MongoDB manager for agent data"""
    
    def __init__(self, connection_string: str, database_name: str):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]
        self.agents_collection = self.db.agents
        self.conversations_collection = self.db.conversations
        self.tasks_collection = self.db.tasks
    
    async def create_agent(self, agent_data: Dict) -> str:
        """Create new agent document"""
        agent_data["created_at"] = datetime.utcnow()
        agent_data["is_active"] = True
        
        result = await self.agents_collection.insert_one(agent_data)
        return str(result.inserted_id)
    
    async def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get agent by ID"""
        return await self.agents_collection.find_one({"agent_id": agent_id})
    
    async def save_conversation(self, conversation_data: Dict) -> str:
        """Save conversation"""
        conversation_data["timestamp"] = datetime.utcnow()
        result = await self.conversations_collection.insert_one(conversation_data)
        return str(result.inserted_id)
    
    async def get_conversation_history(self, agent_id: str, limit: int = 100) -> List[Dict]:
        """Get conversation history"""
        cursor = self.conversations_collection.find(
            {"agent_id": agent_id}
        ).sort("timestamp", -1).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def save_task_result(self, task_data: Dict) -> str:
        """Save task execution result"""
        task_data["created_at"] = datetime.utcnow()
        result = await self.tasks_collection.insert_one(task_data)
        return str(result.inserted_id)
    
    async def close(self):
        """Close database connection"""
        self.client.close()

# MongoDB-backed agent
class MongoAgent(BaseAgent):
    """Agent with MongoDB persistence"""
    
    def __init__(self, agent_id: str, name: str, mongo_manager: MongoDBManager):
        super().__init__(name, ["mongodb_persistence"])
        self.agent_id = agent_id
        self.mongo = mongo_manager
    
    async def initialize(self):
        """Initialize agent in MongoDB"""
        existing = await self.mongo.get_agent(self.agent_id)
        if not existing:
            await self.mongo.create_agent({
                "agent_id": self.agent_id,
                "name": self.name,
                "capabilities": self.capabilities
            })
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """Process message with MongoDB persistence"""
        response = f"MongoDB agent response to: {message}"
        
        # Save conversation
        await self.mongo.save_conversation({
            "agent_id": self.agent_id,
            "user_message": message,
            "agent_response": response,
            "context": context
        })
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "storage": "mongodb",
            "is_active": self.is_active
        }
```

## 📦 Package Management and Environment

### Virtual Environments and Dependencies

**Requirements Management**:
```bash
# Create virtual environment
python -m venv llm-mcp-env

# Activate (Windows)
llm-mcp-env\Scripts\activate
# Activate (Linux/Mac)
source llm-mcp-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate requirements
pip freeze > requirements.txt
```

**requirements.txt for LLM/MCP Development**:
```text
# Core Python packages
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Deep Learning
torch>=2.0.0
transformers>=4.20.0
accelerate>=0.20.0
datasets>=2.0.0

# Web Development
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
websockets>=10.0
aiohttp>=3.8.0
pydantic>=2.0.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0  # PostgreSQL
motor>=3.0.0  # MongoDB async driver
redis>=4.0.0
alembic>=1.8.0  # Database migrations

# Agent Frameworks
langchain>=0.1.0
llama-index>=0.9.0
autogen-agentchat>=0.2.0

# Utilities
python-dotenv>=1.0.0
click>=8.0.0
rich>=13.0.0  # Beautiful terminal output
loguru>=0.7.0  # Better logging

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
httpx>=0.24.0  # For testing FastAPI

# Development
black>=23.0.0  # Code formatting
isort>=5.12.0  # Import sorting
flake8>=6.0.0  # Linting
mypy>=1.0.0   # Type checking
```

**Environment Configuration**:
```python
# config.py
from pydantic import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Keys
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Database
    database_url: str = "postgresql://localhost/agent_db"
    mongodb_url: str = "mongodb://localhost:27017"
    redis_url: str = "redis://localhost:6379"
    
    # Application
    debug: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    
    # Agent Configuration
    default_model: str = "gpt-3.5-turbo"
    max_conversation_history: int = 100
    agent_timeout: float = 30.0
    
    # Security
    secret_key: str = "your-secret-key"
    allowed_origins: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# .env file example
"""
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_API_KEY=your_hf_key
DATABASE_URL=postgresql://user:pass@localhost/agent_db
DEBUG=False
LOG_LEVEL=INFO
"""
```

## 🧪 Testing Strategies

### Unit Testing for Agents

**pytest Configuration**:
```python
# conftest.py
import pytest
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

# Test database setup
TEST_DATABASE_URL = "sqlite:///./test.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_db():
    """Create test database session"""
    Base.metadata.create_all(bind=test_engine)
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    mock = AsyncMock()
    mock.generate_response.return_value = "Test response"
    return mock

@pytest.fixture
def test_agent():
    """Create test agent"""
    return LLMAgent("test_agent", "test-model", "Test system prompt")

@pytest.fixture
def client():
    """Create test client for API testing"""
    return TestClient(app)
```

**Agent Unit Tests**:
```python
# test_agents.py
import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

@pytest.mark.asyncio
async def test_agent_basic_functionality(test_agent):
    """Test basic agent functionality"""
    # Test initialization
    assert test_agent.name == "test_agent"
    assert test_agent.model_name == "test-model"
    assert not test_agent.is_active
    
    # Test memory operations
    test_agent.add_memory("test_key", "test_value")
    assert test_agent.recall_memory("test_key") == "test_value"
    assert test_agent.recall_memory("nonexistent_key") is None

@pytest.mark.asyncio
async def test_agent_message_processing(test_agent, mock_llm_client):
    """Test agent message processing"""
    with patch.object(test_agent, '_generate_response', mock_llm_client.generate_response):
        response = await test_agent.process_message("Hello", {})
        assert response == "Test response"
        assert len(test_agent.conversation_history) == 2  # User + Assistant message

@pytest.mark.asyncio
async def test_multiple_agents_coordination():
    """Test multi-agent coordination"""
    agent1 = LLMAgent("agent1", "model1", "Prompt1")
    agent2 = LLMAgent("agent2", "model2", "Prompt2")
    
    # Mock response generation
    async def mock_response(message, context):
        return f"Response from {agent1.name if 'agent1' in str(agent1) else agent2.name}"
    
    with patch.object(agent1, '_generate_response', mock_response):
        with patch.object(agent2, '_generate_response', mock_response):
            responses = await coordinate_agents([agent1, agent2], "Test task")
            assert len(responses) == 2

@pytest.mark.asyncio
async def test_database_agent_persistence(test_db):
    """Test database agent persistence"""
    agent = DatabaseAgent("test_db_agent", "Test DB Agent", "test-model", "Test prompt", test_db)
    
    # Test message processing with persistence
    response = await agent.process_message("Test message", {"test": "context"})
    assert response is not None
    
    # Check conversation was saved
    conversations = agent.repo.get_conversation_history("test_db_agent")
    assert len(conversations) == 1
    assert conversations[0].user_message == "Test message"
    
    # Test memory persistence
    agent.add_memory("persistent_key", "persistent_value")
    retrieved_value = agent.recall_memory("persistent_key")
    assert retrieved_value == "persistent_value"

def test_rate_limiting(test_agent):
    """Test rate limiting decorator"""
    @rate_limit(max_calls=2, time_window=1)
    def limited_function():
        return "success"
    
    # Should succeed for first 2 calls
    assert limited_function() == "success"
    assert limited_function() == "success"
    
    # Should fail on third call
    with pytest.raises(Exception, match="Rate limit exceeded"):
        limited_function()
```

**API Integration Tests**:
```python
# test_api.py
import pytest
from httpx import AsyncClient
import json

@pytest.mark.asyncio
async def test_agent_message_endpoint():
    """Test agent message API endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Create test agent first
        agent = LLMAgent("api_test_agent", "test-model", "Test prompt")
        app.state.agents = {"api_test_agent": agent}
        
        response = await client.post(
            "/agents/api_test_agent/message",
            json={"content": "Hello API", "context": {}}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "api_test_agent"
        assert "response" in data

@pytest.mark.asyncio
async def test_task_execution_endpoint():
    """Test task execution API endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/tasks/execute",
            json={
                "task": "Test task",
                "agents": ["agent_1", "agent_2"],
                "parallel": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "started"

@pytest.mark.asyncio
async def test_websocket_communication():
    """Test WebSocket communication"""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/agents/test_client") as websocket:
            # Send message
            websocket.send_json({
                "type": "agent_message",
                "agent_id": "agent_1",
                "content": "WebSocket test"
            })
            
            # Receive response
            data = websocket.receive_json()
            assert data["type"] == "agent_response"
            assert data["agent_id"] == "agent_1"
```

## 📚 Essential Libraries and Frameworks

### LLM Integration Libraries

**OpenAI Integration**:
```python
import openai
from typing import List, Dict, Optional
import asyncio

class OpenAIClient:
    """Async OpenAI client wrapper"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None) -> str:
        """Generate response using OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise
    
    async def generate_streaming_response(self, messages: List[Dict[str, str]]):
        """Generate streaming response"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"OpenAI streaming error: {e}")
            raise

# Enhanced LLM Agent with OpenAI
class OpenAIAgent(BaseAgent):
    """Agent powered by OpenAI models"""
    
    def __init__(self, name: str, api_key: str, model: str = "gpt-3.5-turbo",
                 system_prompt: str = "You are a helpful assistant."):
        super().__init__(name, ["text_generation", "conversation", "reasoning"])
        self.client = OpenAIClient(api_key, model)
        self.system_prompt = system_prompt
        self.conversation_history = []
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """Process message using OpenAI"""
        # Build message list
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": message})
        
        # Generate response
        response = await self.client.generate_response(messages)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Limit conversation history length
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.client.model,
            "conversation_length": len(self.conversation_history),
            "capabilities": self.capabilities
        }
```

**Hugging Face Integration**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict

class HuggingFaceClient:
    """Local Hugging Face model client"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def generate_response(self, input_text: str, max_length: int = 512) -> str:
        """Generate response using local model"""
        # Encode input
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input from response
        response = response[len(input_text):].strip()
        return response

class HuggingFaceAgent(BaseAgent):
    """Agent using local Hugging Face models"""
    
    def __init__(self, name: str, model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__(name, ["text_generation", "local_processing"])
        self.client = HuggingFaceClient(model_name)
        self.conversation_history = ""
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """Process message using local model"""
        # Build conversation context
        conversation_input = f"{self.conversation_history}User: {message}\nAssistant:"
        
        # Generate response (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            self.client.generate_response, 
            conversation_input
        )
        
        # Update conversation history
        self.conversation_history += f"User: {message}\nAssistant: {response}\n"
        
        # Limit history length
        if len(self.conversation_history) > 2000:
            # Keep last 1000 characters
            self.conversation_history = self.conversation_history[-1000:]
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.client.model_name,
            "device": self.client.device,
            "conversation_length": len(self.conversation_history)
        }
```

## 🎯 Practical Exercises

### Exercise 1: Build a Multi-Agent Chat System

Create a simple multi-agent chat system where agents can communicate with each other:

```python
# Your implementation here
class MultiAgentChat:
    def __init__(self):
        self.agents = {}
        self.message_broker = MessageBroker()
    
    def add_agent(self, agent: BaseAgent):
        # Implement agent registration
        pass
    
    async def send_message(self, from_agent: str, to_agent: str, message: str):
        # Implement message routing
        pass
    
    async def broadcast_message(self, from_agent: str, message: str):
        # Implement broadcasting
        pass

# Test the system with multiple agents
```

### Exercise 2: Database-Backed Agent Memory

Implement persistent memory for agents using your preferred database:

```python
class PersistentMemoryAgent(BaseAgent):
    """Agent with long-term memory persistence"""
    
    def __init__(self, name: str, db_connection):
        super().__init__(name, ["persistent_memory"])
        self.db = db_connection
    
    async def learn(self, topic: str, information: str):
        """Store learning in long-term memory"""
        # Your implementation
        pass
    
    async def recall(self, topic: str) -> Optional[str]:
        """Recall information from long-term memory"""
        # Your implementation
        pass
    
    async def forget(self, topic: str):
        """Remove information from memory"""
        # Your implementation
        pass
```

### Exercise 3: Agent Performance Monitoring

Build a monitoring system for agent performance:

```python
class AgentMonitor:
    """Monitor agent performance and health"""
    
    def __init__(self):
        self.metrics = {}
    
    def record_response_time(self, agent_id: str, response_time: float):
        # Record response time metrics
        pass
    
    def record_error(self, agent_id: str, error: Exception):
        # Record error metrics
        pass
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        # Return performance statistics
        pass
    
    def generate_report(self) -> Dict[str, Any]:
        # Generate performance report
        pass
```

## ✅ Knowledge Check

Ensure you can:

1. **Implement async agent classes** with proper inheritance
2. **Build RESTful APIs** using FastAPI for agent communication
3. **Integrate databases** for persistent agent memory
4. **Handle real-time communication** using WebSockets
5. **Write comprehensive tests** for agent functionality
6. **Manage environments** and dependencies properly
7. **Integrate with LLM APIs** (OpenAI, Hugging Face, etc.)

## 🚀 Next Steps

With solid programming foundations, continue to:

1. **[Deep Learning Basics](deep-learning.md)** - Neural network implementations
2. **[LLM Architecture](../llms/architecture.md)** - Understanding transformer models
3. **[Building LLM Agents](../agents/architecture.md)** - Advanced agent architectures

---

*Programming skills are the foundation for implementing sophisticated LLM agents and multi-agent systems. Master these concepts before advancing to more complex architectures.*
