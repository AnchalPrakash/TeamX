// Import transformers.js from CDN as an ES Module
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';

// Configure PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

// Disable local models to fetch weights strictly from the Hugging Face CDN
env.allowLocalModels = false;

// DOM Elements & Globals
const jdUpload = document.getElementById('jd-upload');
const resumeUpload = document.getElementById('resume-upload');
const rankBtn = document.getElementById('rank-btn');
const statusMessage = document.getElementById('status-message');
const resultsTableContainer = document.getElementById('results-table-container');
const explanationsContainer = document.getElementById('explanations-container');

// Q&A DOM Elements
const questionInput = document.getElementById('recruiter-question');
const askBtn = document.getElementById('ask-btn');
const answerContainer = document.getElementById('answer-container');
const answerText = document.getElementById('answer-text');

let currentRankedResults = []; // Stores the latest ranking to be queried by the Q&A feature

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

// --- 2. KEYWORD MATCHING LOGIC WITH ALIAS EXPANSION ---
function extractRequiredSkills(jdText) {
    const sectionRegex = /(?:Required\s+Skills|Technical\s+Skills|Skills)\s*:?\s*\n([\s\S]*?)(?:\n\s*\n|$)/i;
    const match = jdText.match(sectionRegex);
    if (!match || !match[1]) return [];
    
    return match[1].split(/[\n•\-*]+/)
        .map(skill => skill.trim())
        .filter(skill => skill.length > 1 && skill.length < 60);
}

// Maps broad tech categories to specific frameworks/tools (solves the hackathon constraint!)
const techSynonyms = {
    "node": ["express", "expressjs", "nestjs", "nodejs"],
    "react": ["reactjs", "nextjs", "react native", "next.js"],
    "javascript": ["js", "es6", "typescript", "ts"],
    "python": ["django", "flask", "fastapi", "pandas"],
    "database": ["sql", "mongodb", "postgres", "postgresql", "mysql", "nosql"],
    "frontend": ["html", "css", "vue", "angular", "ui", "ux"],
    "backend": ["api", "rest", "graphql", "server", "microservices"],
    "versioncontrol": ["git", "github", "gitlab", "bitbucket"],
    "aws": ["amazon web services", "ec2", "s3", "lambda", "cloud"]
};

function keywordScore(resumeText, requiredSkills) {
    if (!requiredSkills || requiredSkills.length === 0) {
        return { score: 0, matched: [], missing: [] };
    }

    const normalize = (str) => str.toLowerCase().replace(/[\W_]+/g, '');
    const lowerResume = resumeText.toLowerCase(); 
    const normalizedResume = normalize(resumeText);
    
    const matched = [];
    const missing = [];
    
    requiredSkills.forEach(skill => {
        const normalizedSkill = normalize(skill);
        if (normalizedSkill.length === 0) return;
        
        let isMatch = false;

        // 1. Literal Exact Match
        if (normalizedResume.includes(normalizedSkill)) {
            isMatch = true;
        } else {
            // 2. Synonym / Related Term Expansion
            for (const [broadCategory, relatedTerms] of Object.entries(techSynonyms)) {
                if (normalizedSkill.includes(broadCategory) || relatedTerms.some(term => normalize(term) === normalizedSkill)) {
                    const foundRelated = relatedTerms.find(term => lowerResume.includes(term));
                    if (foundRelated) {
                        isMatch = true;
                        break;
                    }
                }
            }
        }
        
        if (isMatch) {
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

// --- 5. Q&A LOGIC ---
function answerRecruiterQuestion(question, rankedResults) {
    if (!rankedResults || rankedResults.length === 0) {
        return "Please upload and rank the candidates first before asking questions.";
    }

    const regex = /(?:why did|why is) (.+?) rank(?:ed)? (?:above|higher than|better than|over) (.+?)(?:\?|$)/i;
    const match = question.match(regex);

    if (!match) {
        return "I can currently answer direct comparison questions. Please try formatting like: <em>'Why is [Resume A] ranked above [Resume B]?'</em>";
    }

    const rawName1 = match[1].toLowerCase().trim().replace('.pdf', '');
    const rawName2 = match[2].toLowerCase().trim().replace('.pdf', '');

    const cand1 = rankedResults.find(r => r.filename.toLowerCase().includes(rawName1));
    const cand2 = rankedResults.find(r => r.filename.toLowerCase().includes(rawName2));

    if (!cand1 || !cand2) {
        return `I couldn't find exact matches for those candidates in the current ranking. Please use their filenames. (Looked for "${rawName1}" and "${rawName2}")`;
    }

    if (cand1.finalScore < cand2.finalScore) {
        return `Actually, <strong>${cand2.filename}</strong> is ranked higher than <strong>${cand1.filename}</strong>!`;
    }

    const score1 = (cand1.finalScore * 100).toFixed(1);
    const score2 = (cand2.finalScore * 100).toFixed(1);
    const ai1 = (cand1.semanticScore * 100).toFixed(1);
    const ai2 = (cand2.semanticScore * 100).toFixed(1);

    let answer = `<strong>${cand1.filename}</strong> achieved a higher overall score (${score1}% vs ${score2}%).<br><br>`;

    const matched1 = cand1.matched ? cand1.matched.length : 0;
    const matched2 = cand2.matched ? cand2.matched.length : 0;

    if (matched1 > matched2) {
        answer += `<strong>Keyword Matches:</strong> ${cand1.filename} explicitly matched more required skills (${matched1} vs ${matched2}). `;
        const uniqueTo1 = cand1.matched.filter(s => !cand2.matched.includes(s));
        if (uniqueTo1.length > 0) {
            answer += `Specifically, they had exact matches for <em>${uniqueTo1.join(', ')}</em> which ${cand2.filename} missed. <br><br>`;
        }
    } else if (matched2 > matched1) {
        answer += `<strong>Keyword Matches:</strong> Interestingly, ${cand2.filename} actually matched more explicit keywords (${matched2} vs ${matched1}), but ${cand1.filename}'s contextual alignment pulled them ahead. <br><br>`;
    } else if (matched1 > 0 && matched1 === matched2) {
        answer += `<strong>Keyword Matches:</strong> Both candidates matched the exact same number of required skills (${matched1}). <br><br>`;
    } else {
        answer += `<strong>Keyword Matches:</strong> Neither candidate had explicit keyword matches (or no strict "Required Skills" list was found in the JD format). Therefore, this ranking was decided entirely by the Semantic AI understanding the context of their experience. <br><br>`;
    }

    if (cand1.semanticScore > cand2.semanticScore) {
        answer += `<strong>Semantic AI Score:</strong> Our AI determined that ${cand1.filename}'s overall experience and context aligned better with the Job Description (${ai1}% vs ${ai2}% semantic match).`;
    } else {
        answer += `<strong>Semantic AI Score:</strong> ${cand2.filename} had a slightly better semantic match (${ai2}% vs ${ai1}%), but ${cand1.filename}'s exact keyword matches gave them the overall lead.`;
    }

    return answer;
}

// --- 6. EVENT LISTENERS ---

// Ask Button Logic
if(askBtn) {
    askBtn.addEventListener('click', () => {
        const question = questionInput.value.trim();
        if (!question) return;

        askBtn.textContent = "Analyzing...";
        answerContainer.style.display = 'none';

        setTimeout(() => {
            const answerHtml = answerRecruiterQuestion(question, currentRankedResults);
            answerText.innerHTML = answerHtml;
            answerContainer.style.display = 'block';
            askBtn.textContent = "Ask AI";
        }, 400); 
    });
}

// Rank Button Logic
rankBtn.addEventListener('click', async () => {
    const jdFile = jdUpload.files[0];
    const resumeFiles = resumeUpload.files;

    if (!jdFile) return alert("Please upload a Job Description PDF.");
    if (resumeFiles.length === 0) return alert("Please upload at least one Resume PDF.");

    try {
        rankBtn.disabled = true;
        rankBtn.textContent = "Processing...";
        statusMessage.style.color = "var(--text-muted)";
        statusMessage.textContent = "Status: Extracting text from PDFs...";
        resultsTableContainer.innerHTML = '';
        explanationsContainer.innerHTML = '';

        const jdText = await extractTextFromPDF(jdFile);
        if (!jdText) throw new Error("Could not extract text from the JD file.");

        const resumesDataMap = await extractTextsFromFiles(resumeFiles);
        if (Object.keys(resumesDataMap).length === 0) {
            throw new Error("Could not extract text from any of the uploaded resumes.");
        }

        const extractedSkills = extractRequiredSkills(jdText);

        statusMessage.textContent = "Status: Generating embeddings and scoring candidates...";
        
        // Save to the global variable for the Q&A feature
        currentRankedResults = await rankResumes(jdText, extractedSkills, resumesDataMap);

        let tableHTML = `
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em;">
                <thead>
                    <tr style="background-color: var(--bg-light); text-align: left;">
                        <th style="padding: 12px; border-bottom: 2px solid var(--border-light);">Rank</th>
                        <th style="padding: 12px; border-bottom: 2px solid var(--border-light);">Candidate Resume</th>
                        <th style="padding: 12px; border-bottom: 2px solid var(--border-light);">Final Score</th>
                        <th style="padding: 12px; border-bottom: 2px solid var(--border-light);">Keyword Score</th>
                        <th style="padding: 12px; border-bottom: 2px solid var(--border-light);">Semantic Score</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        currentRankedResults.forEach((candidate, index) => {
            const finalPct = (candidate.finalScore * 100).toFixed(1);
            const keywordPct = candidate.keywordScore !== undefined ? (candidate.keywordScore * 100).toFixed(1) + '%' : 'N/A';
            const semanticPct = (candidate.semanticScore * 100).toFixed(1);
            
            tableHTML += `
                <tr style="border-bottom: 1px solid var(--border-light); transition: background-color 0.2s;">
                    <td style="padding: 12px; font-weight: bold; color: var(--dark-slate);">#${index + 1}</td>
                    <td style="padding: 12px; word-break: break-all; color: var(--dark-slate);">${candidate.filename}</td>
                    <td style="padding: 12px; color: var(--brand-green); font-weight: 700;">${finalPct}%</td>
                    <td style="padding: 12px; color: #64748b;">${keywordPct}</td>
                    <td style="padding: 12px; color: #64748b;">${semanticPct}%</td>
                </tr>
            `;
        });
        tableHTML += `</tbody></table>`;
        resultsTableContainer.innerHTML = tableHTML;

        const top3 = currentRankedResults.slice(0, 3);
        let explanationsHTML = `<div style="display: flex; flex-direction: column; gap: 15px;">`;
        
        top3.forEach((candidate, index) => {
            const dynamicExplanation = generateExplanation(candidate);
            
            explanationsHTML += `
                <div style="background: var(--bg-light); border-left: 4px solid var(--brand-green); padding: 15px; border-radius: 4px;">
                    <h3 style="margin: 0 0 8px 0; font-size: 1.1em; color: var(--dark-slate);">
                        Rank #${index + 1}: ${candidate.filename}
                    </h3>
                    <div style="font-size: 0.9em; color: #475569;">
                        <p style="margin: 0 0 10px 0; color: var(--dark-slate); line-height: 1.5;">${dynamicExplanation}</p>
                        <p style="margin: 8px 0 0 0; background: #fff; padding: 10px; border: 1px dashed var(--border-light); border-radius: 4px; color: #64748b;">
                            <strong>Snippet:</strong> "${candidate.textExcerpt}"
                        </p>
                    </div>
                </div>
            `;
        });
        explanationsHTML += `</div>`;
        explanationsContainer.innerHTML = explanationsHTML;

        statusMessage.style.color = "var(--brand-green)";
        statusMessage.textContent = "Status: Ranking complete!";
        
    } catch (error) {
        console.error("Ranking Error:", error);
        statusMessage.style.color = "red";
        statusMessage.textContent = `Error: ${error.message}`;
    } finally {
        rankBtn.disabled = false;
        rankBtn.textContent = "Let's Go! Rank Resumes";
    }
});
