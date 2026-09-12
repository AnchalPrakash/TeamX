// Import transformers.js from CDN as an ES Module
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';

// Configure PDF.js worker (Required for parsing PDFs properly)
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

// Disable local models to fetch weights strictly from the Hugging Face CDN
env.allowLocalModels = false;

// DOM Elements
const jdUpload = document.getElementById('jd-upload');
const resumeUpload = document.getElementById('resume-upload');
const rankBtn = document.getElementById('rank-btn');
const statusMessage = document.getElementById('status-message');
const resultsTableContainer = document.getElementById('results-table-container');
const explanationsContainer = document.getElementById('explanations-container');

// Helper function to extract text from a PDF file using PDF.js
async function extractTextFromPDF(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    let fullText = '';
    
    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
        const page = await pdf.getPage(pageNum);
        const textContent = await page.getTextContent();
        const pageText = textContent.items.map(item => item.str).join(' ');
        fullText += pageText + ' ';
    }
    
    return fullText.trim();
}

// Main event listener for ranking candidates
rankBtn.addEventListener('click', async () => {
    const jdFile = jdUpload.files[0];
    const resumeFiles = Array.from(resumeUpload.files);

    // Validation
    if (!jdFile) {
        alert("Please upload a Job Description PDF.");
        return;
    }
    if (resumeFiles.length === 0) {
        alert("Please upload at least one Resume PDF.");
        return;
    }

    try {
        rankBtn.disabled = true;
        statusMessage.textContent = "Status: Loading AI model for embeddings... (This might take a moment on first run)";
        
        // 1. Initialize Transformers.js pipeline (using a lightweight feature-extraction model)
        const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        
        statusMessage.textContent = "Status: Parsing PDFs and extracting text...";

        // 2. Extract text from the JD
        const jdText = await extractTextFromPDF(jdFile);
        
        // 3. Extract text from all Resumes
        const resumesData = await Promise.all(resumeFiles.map(async (file) => {
            const text = await extractTextFromPDF(file);
            return { name: file.name, text };
        }));

        statusMessage.textContent = "Status: Generating embeddings and computing similarities...";

        // 4. Generate embeddings for JD (Placeholder logic to be expanded)
        // const jdEmbedding = await extractor(jdText, { pooling: 'mean', normalize: true });
        
        // 5. Generate embeddings for Resumes and calculate cosine similarity
        /* 
           const scoredResumes = await Promise.all(resumesData.map(async (resume) => {
               const resumeEmbedding = await extractor(resume.text, { pooling: 'mean', normalize: true });
               const score = calculateCosineSimilarity(jdEmbedding.data, resumeEmbedding.data);
               return { name: resume.name, score };
           }));
           
           // Sort resumes by descending score
           scoredResumes.sort((a, b) => b.score - a.score);
        */

        // 6. Update UI with Results Table
        resultsTableContainer.innerHTML = `
            <p><em>(Embeddings calculation logic placeholder reached)</em></p>
            <ul>
               <li><strong>JD Parsed:</strong> ${jdText.substring(0, 100)}...</li>
               <li><strong>Total Resumes Parsed:</strong> ${resumesData.length}</li>
            </ul>
        `;

        // 7. Update UI with Explanations
        explanationsContainer.innerHTML = `
            <p>Ready to generate top 3 explanations based on cosine similarity scores.</p>
        `;

        statusMessage.textContent = "Status: Complete!";
        
    } catch (error) {
        console.error(error);
        statusMessage.textContent = `Error: ${error.message}`;
    } finally {
        rankBtn.disabled = false;
    }
});