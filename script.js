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
            if (text) {
                textsMap[file.name] = text;
            } else {
                console.warn(`Warning: "${file.name}" was parsed but returned no text.`);
            }
        } catch (error) {
            console.warn(`Warning: Skipped unreadable or corrupt file "${file.name}".`, error);
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
    
    const rawSkills = match[1].split(/[\n•\-*]+/);
    return rawSkills
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

// Cache for the pipeline and text embeddings to optimize loop performance
let extractorPipeline = null;
const embeddingCache = new Map();

/**
 * Returns a mean-pooled, normalized embedding vector for the provided text.
 * Loads the model on the first run and shows a loading indicator.
 */
async function getEmbedding(text) {
    // Return cached vector if this exact text was already processed (e.g., the JD text)
    if (embeddingCache.has(text)) {
        return embeddingCache.get(text);
    }

    if (!extractorPipeline) {
        statusMessage.textContent = "Status: Downloading/Initializing AI model... (This takes a few seconds on first run)";
        extractorPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        statusMessage.textContent = "Status: AI Model loaded successfully.";
    }

    const output = await extractorPipeline(text, { pooling: 'mean', normalize: true });
    const vector = Array.from(output.data);
    
    embeddingCache.set(text, vector);
    return vector;
}

/**
 * Calculates the cosine similarity between two vectors.
 */
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Embeds both texts and returns their cosine similarity as a 0-1 score.
 */
async function semanticScore(jdText, resumeText) {
    const jdEmbed = await getEmbedding(jdText);
    const resumeEmbed = await getEmbedding(resumeText);
    return cosineSimilarity(jdEmbed, resumeEmbed);
}

// --- 4. MAIN UI EVENT LISTENER ---

rankBtn.addEventListener('click', async () => {
    const jdFile = jdUpload.files[0];
    const resumeFiles = resumeUpload.files;

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
        statusMessage.style.color = "var(--text-muted)";
        statusMessage.textContent = "Status: Parsing PDFs...";
        resultsTableContainer.innerHTML = '';
        explanationsContainer.innerHTML = '';

        const jdText = await extractTextFromPDF(jdFile);
        if (!jdText) throw new Error("Could not extract text from the JD file.");

        const extractedSkills = extractRequiredSkills(jdText);
        
        const resumesDataMap = await extractTextsFromFiles(resumeFiles);
        const validResumeNames = Object.keys(resumesDataMap);
        
        if (validResumeNames.length === 0) {
            throw new Error("Could not extract text from any of the uploaded resumes.");
        }

        statusMessage.textContent = "Status: Scoring Candidates against JD...";
        const scoredResumes = [];
        
        for (const name of validResumeNames) {
            const text = resumesDataMap[name];
            
            // 1. Calculate Semantic Score using the requested function
            const aiScore = await semanticScore(jdText, text);
            
            // 2. Calculate Keyword Score
            const keywordData = keywordScore(text, extractedSkills);
            
            // 3. Hybrid Score (50/50 if skills exist, otherwise 100% AI)
            let finalScore = aiScore;
            if (extractedSkills.length > 0) {
                finalScore = (aiScore + keywordData.score) / 2;
            }

            scoredResumes.push({ 
                name, 
                finalScore,
                aiScore,
                keywordData,
                textExcerpt: text.substring(0, 150) + "..." 
            });
        }

        // Sort highest score first
        scoredResumes.sort((a, b) => b.finalScore - a.finalScore);

        // Build Results Table
        let tableHTML = `
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.95em;">
                <thead>
                    <tr style="background-color: #f9fafb; text-align: left;">
                        <th style="padding: 12px; border-bottom: 2px solid #e5e7eb;">Rank</th>
                        <th style="padding: 12px; border-bottom: 2px solid #e5e7eb;">Candidate Resume</th>
                        <th style="padding: 12px; border-bottom: 2px solid #e5e7eb;">Match Score</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        scoredResumes.forEach((candidate, index) => {
            const percentage = (candidate.finalScore * 100).toFixed(2);
            tableHTML += `
                <tr style="border-bottom: 1px solid #e5e7eb;">
                    <td style="padding: 12px;"><strong>#${index + 1}</strong></td>
                    <td style="padding: 12px; word-break: break-all;">${candidate.name}</td>
                    <td style="padding: 12px; color: #2563eb; font-weight: 600;">${percentage}%</td>
                </tr>
            `;
        });
        tableHTML += `</tbody></table>`;
        resultsTableContainer.innerHTML = tableHTML;

        // Build Top 3 Explanations
        const top3 = scoredResumes.slice(0, 3);
        let explanationsHTML = `<div style="display: flex; flex-direction: column; gap: 15px;">`;
        
        top3.forEach((candidate, index) => {
            const finalPercentage = (candidate.finalScore * 100).toFixed(1);
            const aiPercentage = (candidate.aiScore * 100).toFixed(1);
            const keywordPercentage = (candidate.keywordData.score * 100).toFixed(1);
            
            const matchedTags = candidate.keywordData.matched.map(skill => `<span style="display: inline-block; background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin: 2px;">✓ ${skill}</span>`).join('');
            const missingTags = candidate.keywordData.missing.map(skill => `<span style="display: inline-block; background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin: 2px;">✕ ${skill}</span>`).join('');

            explanationsHTML += `
                <div style="background: #f8fafc; border-left: 4px solid #2563eb; padding: 15px; border-radius: 4px;">
                    <h3 style="margin: 0 0 8px 0; font-size: 1.1em; color: #1e293b;">
                        Rank #${index + 1}: ${candidate.name}
                    </h3>
                    <div style="font-size: 0.9em; color: #475569;">
                        <p style="margin: 0 0 8px 0;"><strong>Overall Score:</strong> ${finalPercentage}% (AI Semantic Match: ${aiPercentage}% | Exact Keyword Match: ${extractedSkills.length > 0 ? keywordPercentage + '%' : 'N/A'})</p>
                        
                        ${extractedSkills.length > 0 ? `
                        <div style="margin-bottom: 8px;">
                            <strong>Matched Skills:</strong> ${matchedTags || '<em>None</em>'}
                        </div>
                        <div style="margin-bottom: 8px;">
                            <strong>Missing Skills:</strong> ${missingTags || '<em>None</em>'}
                        </div>
                        ` : '<p style="color: #d97706; margin-bottom: 8px;"><em>Could not parse a strict "Required Skills" list from the JD format. Score based entirely on AI semantic understanding.</em></p>'}
                        
                        <p style="margin: 8px 0 0 0; background: #fff; padding: 10px; border: 1px dashed #cbd5e1; border-radius: 4px;">
                            <strong>Resume Snippet:</strong> "${candidate.textExcerpt}"
                        </p>
                    </div>
                </div>
            `;
        });
        explanationsHTML += `</div>`;
        explanationsContainer.innerHTML = explanationsHTML;

        statusMessage.textContent = "Status: Ranking complete!";
        
    } catch (error) {
        console.error("Ranking Error:", error);
        statusMessage.style.color = "red";
        statusMessage.textContent = `Error: ${error.message}`;
    } finally {
        rankBtn.disabled = false;
    }
});
