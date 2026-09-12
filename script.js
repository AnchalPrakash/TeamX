// Import transformers.js from CDN as an ES Module
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';

// Configure PDF.js worker
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

// --- 1. PDF EXTRACTION LOGIC ---
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

async function extractTextsFromFiles(fileList) {
    const textsMap = {};
    const filesArray = Array.from(fileList);

    const extractionPromises = filesArray.map(async (file) => {
        try {
            const text = await extractTextFromPDF(file);
            if (text) textsMap[file.name] = text;
        } catch (error) {
            console.warn(`Skipped unreadable file "${file.name}".`, error);
        }
    });

    await Promise.all(extractionPromises);
    return textsMap;
}

// --- 2. KEYWORD MATCHING LOGIC ---
function extractRequiredSkills(jdText) {
    const sectionRegex = /(?:Required\s+Skills|Technical\s+Skills|Skills)\s*:?\s*\n([\s\S]*?)(?:\n\s*\n|$)/i;
    const match = jdText.match(sectionRegex);
    if (!match || !match[1]) return [];
    
    return match[1].split(/[\n•\-*]+/)
        .map(skill => skill.trim())
        .filter(skill => skill.length > 1 && skill.length < 60);
}

function keywordScore(resumeText, requiredSkills) {
    if (!requiredSkills || requiredSkills.length === 0) {
        return { score: 0, matched: [], missing: [] };
    }

    const normalize = (str) => str.toLowerCase().replace(/[\W_]+/g, '');
    const normalizedResume = normalize(resumeText);
    const matched = [];
    const missing = [];
    
    requiredSkills.forEach(skill => {
        const normalizedSkill = normalize(skill);
        if (normalizedSkill.length === 0) return;
        
        if (normalizedResume.includes(normalizedSkill)) {
            matched.push(skill);
        } else {
            missing.push(skill);
        }
    });
    
    const totalValidSkills = matched.length + missing.length;
    const score = totalValidSkills === 0 ? 0 : matched.length / totalValidSkills;
    return { score, matched, missing };
}

// --- 3. AI SEMANTIC EMBEDDING LOGIC ---
let extractorPipeline = null;
const embeddingCache = new Map();

async function getEmbedding(text) {
    if (embeddingCache.has(text)) return embeddingCache.get(text);

    if (!extractorPipeline) {
        statusMessage.textContent = "Status: Downloading/Initializing AI model... (This takes a few seconds on first run)";
        extractorPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }

    const output = await extractorPipeline(text, { pooling: 'mean', normalize: true });
    const vector = Array.from(output.data);
    
    embeddingCache.set(text, vector);
    return vector;
}

function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function semanticScore(jdText, resumeText) {
    const jdEmbed = await getEmbedding(jdText);
    const resumeEmbed = await getEmbedding(resumeText);
    return cosineSimilarity(jdEmbed, resumeEmbed);
}

// --- 4. ORCHESTRATION & EXPLANATION LOGIC ---
async function rankResumes(jdText, requiredSkills, resumeTexts) {
    const scoredResumes = [];
    const filenames = Object.keys(resumeTexts);

    for (const filename of filenames) {
        const text = resumeTexts[filename];
        
        const keywordData = keywordScore(text, requiredSkills);
        const aiScore = await semanticScore(jdText, text);
        
        let finalScore = aiScore;
        if (requiredSkills && requiredSkills.length > 0) {
            finalScore = (0.5 * keywordData.score) + (0.5 * aiScore);
        }
        
        scoredResumes.push({
            filename,
            finalScore,
            keywordScore: keywordData.score,
            semanticScore: aiScore,
            matched: keywordData.matched,
            missing: keywordData.missing,
            textExcerpt: text.substring(0, 150) + "..."
        });
    }
    
    scoredResumes.sort((a, b) => b.finalScore - a.finalScore);
    return scoredResumes;
}

function generateExplanation(entry) {
    const finalPercent = (entry.finalScore * 100).toFixed(1);
    let text = `This candidate achieved a final match score of ${finalPercent}%. `;

    if (!entry.matched || !entry.missing || (entry.matched.length === 0 && entry.missing.length === 0)) {
        return text + `Because no distinct skills were extracted from the JD, this ranking relies entirely on the AI's semantic understanding of their experience.`;
    }

    if (entry.matched.length > 0) {
        const displayMatches = entry.matched.slice(0, 5).join(", ");
        text += `Their resume successfully highlighted key qualifications, including ${displayMatches}. `;
    } else {
        text += `Unfortunately, their resume did not contain explicit matches for the required keywords. `;
    }

    if (entry.missing.length > 0) {
        const displayMissing = entry.missing.slice(0, 5).join(", ");
        text += `However, they appear to be missing certain required skills such as ${displayMissing}.`;
    } else if (entry.matched.length > 0) {
        text += `Impressively, they perfectly matched every explicit skill required for the role.`;
    }

    return text.trim();
}

// --- 5. MAIN UI EVENT LISTENER ---
rankBtn.addEventListener('click', async () => {
    // 1. Read the JD file and resume files from the file inputs
    const jdFile = jdUpload.files[0];
    const resumeFiles = resumeUpload.files;

    if (!jdFile) return alert("Please upload a Job Description PDF.");
    if (resumeFiles.length === 0) return alert("Please upload at least one Resume PDF.");

    try {
        // 7. Show loading state on button and status message
        rankBtn.disabled = true;
        rankBtn.textContent = "Processing...";
        statusMessage.style.color = "var(--text-muted)";
        statusMessage.textContent = "Status: Extracting text from PDFs...";
        resultsTableContainer.innerHTML = '';
        explanationsContainer.innerHTML = '';

        // 2. Extract all text via extractTextFromPDF / extractTextsFromFiles
        const jdText = await extractTextFromPDF(jdFile);
        if (!jdText) throw new Error("Could not extract text from the JD file.");

        const resumesDataMap = await extractTextsFromFiles(resumeFiles);
        if (Object.keys(resumesDataMap).length === 0) {
            throw new Error("Could not extract text from any of the uploaded resumes.");
        }

        // 3. Extract required skills from the JD
        const extractedSkills = extractRequiredSkills(jdText);

        // 4. Run rankResumes()
        statusMessage.textContent = "Status: Generating embeddings and scoring candidates...";
        const scoredResumes = await rankResumes(jdText, extractedSkills, resumesDataMap);

        // 5. Render the full ranked list into the results table
        let tableHTML = `
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em;">
                <thead>
                    <tr style="background-color: #f9fafb; text-align: left;">
                        <th style="padding: 12px; border-bottom: 2px solid #e5e7eb;">Rank</th>
                        <th style="padding: 12px; border-bottom: 2px solid #e5e7eb;">Candidate Resume</th>
                        <th style="padding: 12px; border-bottom: 2px solid #e5e7eb;">Final Score</th>
                        <th style="padding: 12px; border-bottom: 2px solid #e5e7eb;">Keyword Score</th>
                        <th style="padding: 12px; border-bottom: 2px solid #e5e7eb;">Semantic Score</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        scoredResumes.forEach((candidate, index) => {
            const finalPct = (candidate.finalScore * 100).toFixed(1);
            const keywordPct = candidate.keywordScore !== undefined ? (candidate.keywordScore * 100).toFixed(1) + '%' : 'N/A';
            const semanticPct = (candidate.semanticScore * 100).toFixed(1);
            
            tableHTML += `
                <tr style="border-bottom: 1px solid #e5e7eb; transition: background-color 0.2s;">
                    <td style="padding: 12px; font-weight: bold; color: #475569;">#${index + 1}</td>
                    <td style="padding: 12px; word-break: break-all; color: #1e293b;">${candidate.filename}</td>
                    <td style="padding: 12px; color: #2563eb; font-weight: 700;">${finalPct}%</td>
                    <td style="padding: 12px; color: #64748b;">${keywordPct}</td>
                    <td style="padding: 12px; color: #64748b;">${semanticPct}%</td>
                </tr>
            `;
        });
        tableHTML += `</tbody></table>`;
        resultsTableContainer.innerHTML = tableHTML;

        // 6. Render the top 3 with their generateExplanation() text
        const top3 = scoredResumes.slice(0, 3);
        let explanationsHTML = `<div style="display: flex; flex-direction: column; gap: 15px;">`;
        
        top3.forEach((candidate, index) => {
            const dynamicExplanation = generateExplanation(candidate);
            
            explanationsHTML += `
                <div style="background: #f8fafc; border-left: 4px solid #2563eb; padding: 15px; border-radius: 4px;">
                    <h3 style="margin: 0 0 8px 0; font-size: 1.1em; color: #1e293b;">
                        Rank #${index + 1}: ${candidate.filename}
                    </h3>
                    <div style="font-size: 0.9em; color: #475569;">
                        <p style="margin: 0 0 10px 0; color: #334155; line-height: 1.5;">${dynamicExplanation}</p>
                        <p style="margin: 8px 0 0 0; background: #fff; padding: 10px; border: 1px dashed #cbd5e1; border-radius: 4px; color: #64748b;">
                            <strong>Snippet:</strong> "${candidate.textExcerpt}"
                        </p>
                    </div>
                </div>
            `;
        });
        explanationsHTML += `</div>`;
        explanationsContainer.innerHTML = explanationsHTML;

        statusMessage.style.color = "#166534";
        statusMessage.textContent = "Status: Ranking complete!";
        
    } catch (error) {
        console.error("Ranking Error:", error);
        statusMessage.style.color = "red";
        statusMessage.textContent = `Error: ${error.message}`;
    } finally {
        // Reset button state
        rankBtn.disabled = false;
        rankBtn.textContent = "Rank Candidates";
    }
});
