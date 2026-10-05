# 🖥️ Network & Sysadmin Assistant (RAG Chatbot)

Web Application แชตบอตตอบคำถามด้าน IT Infrastructure และการตั้งค่าอุปกรณ์เครือข่าย พัฒนาด้วยเทคนิค RAG (Retrieval-Augmented Generation) เพื่อให้ได้คำตอบที่แม่นยำและอ้างอิงจากคลังความรู้ขององค์กรโดยเฉพาะ

## 📌 แนวคิดของ Domain
โปรเจกต์นี้ถูกออกแบบมาเพื่อทำหน้าที่เป็น **ผู้ช่วยวิศวกรเครือข่าย (Network Engineer) และผู้ดูแลระบบ (System Administrator)** โดยใช้ AI ในการช่วยค้นหาและตอบคำถามเชิงเทคนิคจากคู่มือการปฏิบัติงานได้อย่างรวดเร็ว ไม่ว่าจะเป็น Command Line ของอุปกรณ์ Network (Cisco, Aruba), การตั้งค่าเซิร์ฟเวอร์ Linux (Ubuntu), ระบบ Virtualization (Proxmox, EVE-NG) ไปจนถึงการตรวจสอบข้อกำหนด Policy ในการออกแบบ Topology

## 📚 แหล่งที่มาของเอกสาร
ข้อมูลความรู้ถูกจัดเก็บอยู่ในโฟลเดอร์ `data/` ประกอบด้วยไฟล์นามสกุล `.txt` จำนวน 10 ไฟล์ (ความยาวรวมมากกว่า 15,000 ตัวอักษร) ซึ่งครอบคลุมเนื้อหา 4 หมวดหมู่หลัก:
1. **Network Devices:** คู่มือพื้นฐาน Cisco CLI, การแก้ปัญหาและการทำ Factory Reset สวิตช์ Aruba, และการตั้งค่า OSPF Routing Protocol
2. **Server & Infrastructure:** การทำ High Availability ด้วย Keepalived และการตั้งค่า Centralized Log Server (Syslog) บน Ubuntu
3. **Virtualization & Lab:** คู่มือการตั้งค่า VM บน Proxmox VE และการนำเข้า Image เพื่อเชื่อมต่อ Node ใน EVE-NG
4. **Architecture & Security:** ข้อกำหนด Network Topology (ระบุเงื่อนไขเฉพาะว่าต้องมีเพียง Zone เดียวที่ต่อเข้ากับ Static Zone นอกนั้นให้ต่อเข้าหากันเอง), IPsec VPN, และขั้นตอนการ Diagnostics เบื้องต้น (Ping, Traceroute, ARP)

## 🛠️ เทคโนโลยีที่ใช้ (Tech Stack)
* **Frontend & Deployment:** Streamlit / Streamlit Community Cloud
* **RAG Framework:** LangChain (`langchain`, `langchain-google-genai`, `langchain-community`)
* **Document Processing:** `RecursiveCharacterTextSplitter` (chunk_size=1000, chunk_overlap=150)
* **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (ผ่าน `langchain-huggingface`)
* **Vector Database:** FAISS (`faiss-cpu`)
* **LLM:** Google Gemini API (`gemini-1.5-flash`)

## 🤖 ตัวอย่าง Prompt ที่ใช้สั่ง AI
ระบบใช้ Prompt Engineering ในการควบคุมบทบาทและขอบเขตการตอบของ LLM ดังนี้:

> "คุณคือ Senior Network Engineer จงตอบคำถามหรือให้คำแนะนำทางเทคนิคโดยอ้างอิงจากคู่มือ (Context) ด้านล่างนี้เท่านั้น
> 
> กฎเหล็ก:
> 1. หากข้อมูลใน Context ไม่มีคำตอบ ให้ตอบว่า: 'ไม่พบข้อมูล' ห้ามเดาหรือสร้างคำสั่งขึ้นมาเองเด็ดขาด
> 2. หากตอบได้ ให้แสดงขั้นตอนหรือ Command Line ออกมาให้ชัดเจน พร้อมระบุชื่อไฟล์อ้างอิงตอนท้าย"

## 🚀 วิธีการติดตั้งและรันโปรเจกต์ (Local Development)

1. **โคลนโปรเจกต์และเข้าไปที่โฟลเดอร์**
   ```bash
   git clone <your-github-repo-url>
   cd Network-Sysadmin-Assistant

2. **ติดตั้งไลบรารีที่จำเป็น**
    ```bash
    pip install -r requirements.txt

3. **ตั้งค่า API Key**
    สร้างโฟลเดอร์ .streamlit และไฟล์ secrets.toml ไว้ด้านในสุดของโปรเจกต์ พร้อมเพิ่ม API Key:
    ```Ini, TOML
    GEMINI_API_KEY = "AIza_ใส่คีย์ของคุณที่นี่..."
4. **รันแอปพลิเคชัน**
    ```bash
    python -m streamlit run app.py