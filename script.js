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

/**
 * Extracts all text from a single PDF File object using PDF.js.
 * @param {File} file - The PDF file object.
 * @returns {Promise<string>} - The concatenated text of all pages.
 */
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

/**
 * Extracts text from multiple PDF files, mapping filenames to their text content.
 * Gracefully skips unreadable or corrupt files.
 * @param {FileList|Array<File>} fileList - The list of PDF files to process.
 * @returns {Promise<Object>} - An object mapping { "filename.pdf": "extracted text..." }
 */
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

/**
 * Parses a list of required skills from the Job Description text.
 * @param {string} jdText - The extracted text from the JD PDF.
 * @returns {Array<string>} - An array of cleaned skill strings.
 */
function extractRequiredSkills(jdText) {
    const sectionRegex = /(?:Required\s+Skills|Technical\s+Skills|Skills)\s*:?\s*\n([\s\S]*?)(?:\n\s*\n|$)/i;
    const match = jdText.match(sectionRegex);
    
    if (!match || !match[1]) {
        return [];
    }
    
    const skillsBlock = match[1];
    const rawSkills = skillsBlock.split(/[\n•\-*]+/);
    
    return rawSkills
        .map(skill => skill.trim())
        .filter(skill => skill.length > 1 && skill.length < 60);
}

/**
 * Checks how many required skills appear in the resume text, case-insensitively.
 * @param {string} resumeText - The extracted text from the Resume PDF.
 * @param {Array<string>} requiredSkills - The array of skills extracted from the JD.
 * @returns {Object} - { score (0-1), matched: [...], missing: [...] }
 */
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

/**
 * Calculates the cosine similarity between two numeric vectors.
 */
function calculateCosineSimilarity(vecA, vecB) {
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

// Main event listener for ranking candidates
rankBtn.addEventListener('click', async () => {
    const jdFile = jdUpload.files[0];
    const resumeFiles = resumeUpload.files;

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
        statusMessage.style.color = "var(--text-muted)";
        statusMessage.textContent = "Status: Parsing Job Description and Resumes...";
        resultsTableContainer.innerHTML = '';
        explanationsContainer.innerHTML = '';

        // 1. Extract texts
        const jdText = await extractTextFromPDF(jdFile);
        if (!jdText) {
            throw new Error("Could not extract text from the JD file. It might be scanned or empty.");
        }

        // 1a. Extract exact skills for keyword matching
        const extractedSkills = extractRequiredSkills(jdText);
        console.log("Found JD Skills:", extractedSkills);

        const resumesDataMap = await extractTextsFromFiles(resumeFiles);
        const validResumeNames = Object.keys(resumesDataMap);
        
        if (validResumeNames.length === 0) {
            throw new Error("Could not extract text from any of the uploaded resumes. They might be corrupt or scanned images.");
        }

        // 2. Initialize Transformers.js pipeline
        statusMessage.textContent = "Status: Loading AI model for embeddings... (This takes a moment on the first run)";
        const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

        // 3. Generate JD Embedding
        statusMessage.textContent = "Status: Analyzing Job Description...";
        const jdEmbeddingOutput = await extractor(jdText, { pooling: 'mean', normalize: true });
        const jdEmbedding = jdEmbeddingOutput.data;

        // 4. Generate Resume Embeddings and Calculate Scores
        statusMessage.textContent = "Status: Scoring Candidates against JD...";
        const scoredResumes = [];
        
        for (const name of validResumeNames) {
            const text = resumesDataMap[name];
            
            // Calculate Semantic Score (AI)
            const resumeEmbeddingOutput = await extractor(text, { pooling: 'mean', normalize: true });
            const resumeEmbedding = resumeEmbeddingOutput.data;
            const semanticScore = calculateCosineSimilarity(jdEmbedding, resumeEmbedding);
            
            // Calculate Keyword Score (Exact Match)
            const keywordData = keywordScore(text, extractedSkills);
            
            // Hybrid Score: 50% Semantic AI, 50% Exact Keyword Match 
            // (If no skills were found in JD, rely 100% on Semantic Score)
            let finalScore = semanticScore;
            if (extractedSkills.length > 0) {
                finalScore = (semanticScore + keywordData.score) / 2;
            }

            scoredResumes.push({ 
                name, 
                finalScore,
                semanticScore,
                keywordData,
                textExcerpt: text.substring(0, 150) + "..." 
            });
        }

        // 5. Sort resumes by highest hybrid score first
        scoredResumes.sort((a, b) => b.finalScore - a.finalScore);

        // 6. Build the Results Table HTML
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

        // 7. Build the Top 3 Explanations HTML
        const top3 = scoredResumes.slice(0, 3);
        let explanationsHTML = `<div style="display: flex; flex-direction: column; gap: 15px;">`;
        
        top3.forEach((candidate, index) => {
            const finalPercentage = (candidate.finalScore * 100).toFixed(1);
            const aiPercentage = (candidate.semanticScore * 100).toFixed(1);
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
