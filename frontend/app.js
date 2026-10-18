/**
 * Memorial MindCare AI Platform - Frontend Controller
 * Integrates GSAP GreenSock Animations, Multi-Modal Risk Scoring, Web Audio API Analyzer, FHIR Exporter & Signed Reports.
 */

const API_BASE = "http://localhost:8000";

// Global State
let activeMrn = "MRN-2026-1023";
let patientDataMap = {};
let currentPrediction = null;
let currentXAI = null;
let currentDoctorReport = null;
let currentFhirBundle = null;

// Quiz State
let quizData = {
    sleep_hours: 5.5,
    social_score: 20,
    screen_hrs: 10.0,
    activity_min: 15,
    stress_level: 8
};

// Web Audio API State
let audioCtx = null;
let analyserNode = null;
let micStream = null;
let isRecordingMic = false;
let audioAnimFrame = null;

// Charts
let chartTrend = null;
let chartShap = null;
let gauges = {};

document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initGaugeCharts();
    initTrendChart();
    initShapChart();
    fetchPatients();
    fetchModelMetrics();
    fetchGeminiStatus();

    // Initial GSAP Page Load Entrance Animation
    runGsapEntranceAnimation();

    document.getElementById("patientSelect").addEventListener("change", (e) => {
        activeMrn = e.target.value;
        loadPatientData(activeMrn);
    });
});

// GSAP Animations
function runGsapEntranceAnimation() {
    if (typeof gsap !== "undefined") {
        gsap.from(".hospital-header", { duration: 0.8, y: -40, opacity: 0, ease: "power2.out" });
        gsap.from(".welcome-hero", { duration: 0.8, y: 30, opacity: 0, delay: 0.2, ease: "power2.out" });
        gsap.from(".emr-card", { duration: 0.7, y: 30, opacity: 0, stagger: 0.1, delay: 0.3, ease: "power2.out" });
    }
}

function animateGsapTabContent(tabId) {
    if (typeof gsap !== "undefined") {
        gsap.from(`#${tabId} .emr-card, #${tabId} .risk-card, #${tabId} .quiz-step-card`, {
            duration: 0.5,
            y: 20,
            opacity: 0,
            stagger: 0.08,
            ease: "power2.out"
        });
    }
}

// Counter Roll-Up Animation
function animateCounter(id, targetValue, suffix = "%", duration = 1000) {
    const el = document.getElementById(id);
    if (!el) return;
    
    let startTimestamp = null;
    const startVal = parseFloat(el.innerText.replace(/[^0-9.]/g, "")) || 0;
    const endVal = parseFloat(targetValue);

    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = startVal + (endVal - startVal) * progress;
        el.innerText = `${current.toFixed(1)}${suffix}`;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Theme Management
function initTheme() {
    const themeBtn = document.getElementById("themeToggleBtn");
    themeBtn.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme");
        const nextTheme = currentTheme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", nextTheme);
        themeBtn.innerHTML = nextTheme === "dark" ? '<i class="fa-solid fa-moon"></i>' : '<i class="fa-solid fa-sun"></i>';
        
        if (currentPrediction) renderTrendChart(currentPrediction.historical_trend);
        if (currentXAI) renderShapChart(currentXAI.top_contributing_factors);
    });
}

// Tab Switching
function switchTab(tabName) {
    document.querySelectorAll(".emr-tab").forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("data-tab") === tabName);
    });

    document.querySelectorAll(".tab-pane").forEach(tab => {
        tab.classList.toggle("active", tab.id === `tab-${tabName}`);
    });

    // Run GSAP Animation on Tab Change
    animateGsapTabContent(`tab-${tabName}`);
}

// Gamified Quiz Functions
function nextQuizStep(stepNum) {
    document.querySelectorAll(".quiz-step-card").forEach(card => card.classList.remove("active"));
    const nextCard = document.getElementById(`quizStep${stepNum}`);
    nextCard.classList.add("active");
    
    if (typeof gsap !== "undefined") {
        gsap.from(nextCard, { duration: 0.4, scale: 0.98, opacity: 0, ease: "back.out(1.5)" });
    }

    document.getElementById("quizProgressBar").style.width = `${stepNum * 20}%`;
}

function selectQuizSleep(hours) {
    quizData.sleep_hours = hours;
    event.target.parentElement.querySelectorAll(".btn-quiz-opt").forEach(btn => btn.classList.remove("selected"));
    event.target.classList.add("selected");
}

function selectQuizSocial(socialScore, screenHrs) {
    quizData.social_score = socialScore;
    quizData.screen_hrs = screenHrs;
    event.target.parentElement.querySelectorAll(".btn-quiz-opt").forEach(btn => btn.classList.remove("selected"));
    event.target.classList.add("selected");
}

function selectQuizActivity(activityMin) {
    quizData.activity_min = activityMin;
    event.target.parentElement.querySelectorAll(".btn-quiz-opt").forEach(btn => btn.classList.remove("selected"));
    event.target.classList.add("selected");
}

async function submitGamifiedQuiz() {
    quizData.stress_level = parseInt(document.getElementById("inpQuizStress").value);
    const lifeSentence = document.getElementById("qzSentence").value;
    const hobbies = document.getElementById("qzHobbies").value;

    const payload = {
        patient_id: activeMrn,
        life_sentence: lifeSentence,
        sleep_hours: quizData.sleep_hours,
        social_score: quizData.social_score,
        screen_hrs: quizData.screen_hrs,
        exercise_hobbies: hobbies,
        activity_min: quizData.activity_min,
        stress_level: quizData.stress_level
    };

    try {
        const res = await fetch(`${API_BASE}/api/quiz-intake`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            const data = await res.json();
            const feats = data.extracted_quiz_features;

            populateIntakeForm(feats);
            const patient = patientDataMap[activeMrn] || { id: activeMrn, name: "Intake Patient", age: 25, gender: "Male" };
            await runInference(feats, patient.id, patient.name, patient.age, patient.gender);

            alert("🎮 Gamified Quiz Submitted! Assessment updated with parallel data entry.");
            switchTab("overview");
        }
    } catch (err) {
        console.warn("Quiz intake error:", err);
    }
}

// FHIR R4 Exporter Functions
async function exportFHIRBundle() {
    const patient = patientDataMap[activeMrn] || { id: activeMrn, name: "Vedant", age: 21, gender: "Male" };
    const payload = {
        id: patient.id,
        name: patient.name,
        age: patient.age,
        gender: patient.gender,
        phq9_score: parseFloat(document.getElementById("inpPhq9").value),
        gad7_score: parseFloat(document.getElementById("inpGad7").value),
        mdq_score: parseFloat(document.getElementById("inpMdq").value),
        sleep_hours: parseFloat(document.getElementById("inpSleep").value),
        stress_level: parseInt(document.getElementById("inpStress").value)
    };

    try {
        const res = await fetch(`${API_BASE}/api/export-fhir`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            currentFhirBundle = await res.json();
            document.getElementById("fhirJsonText").value = JSON.stringify(currentFhirBundle, null, 2);
            document.getElementById("fhirModal").classList.remove("hidden");
            if (typeof gsap !== "undefined") {
                gsap.from("#fhirModal .modal-card", { duration: 0.4, scale: 0.9, opacity: 0, ease: "back.out(1.4)" });
            }
        }
    } catch (err) {
        console.warn("FHIR export error:", err);
    }
}

function closeFhirModal() {
    document.getElementById("fhirModal").classList.add("hidden");
}

function downloadFhirJson() {
    if (!currentFhirBundle) return;
    const jsonStr = JSON.stringify(currentFhirBundle, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `FHIR-R4-${activeMrn}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// Multi-Doctor Co-Signing Functions
async function addDoctorCosign() {
    const doctorName = prompt("Enter Second Opinion Physician Name:", "Dr. Amanda Vance, MD");
    if (!doctorName) return;
    const comments = prompt("Enter Clinical Review Comments:", "Reviewed and concurred with primary clinical AI evaluation and preventive plan.");

    try {
        const res = await fetch(`${API_BASE}/api/multi-doctor/cosign`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                patient_id: activeMrn,
                doctor_name: doctorName,
                comments: comments
            })
        });

        if (res.ok) {
            const data = await res.json();
            renderConsultationAuditLogs(data.all_consultation_logs);
            alert(`Co-signature recorded for ${doctorName}!`);
        }
    } catch (err) {
        console.warn("Cosign error:", err);
    }
}

function renderConsultationAuditLogs(logs) {
    const container = document.getElementById("consultationAuditLogs");
    if (!container || !logs) return;
    container.innerHTML = logs.map(l => `
        <div class="log-entry-card">
            <div style="display:flex; justify-content:space-between;">
                <strong><i class="fa-solid fa-user-doctor text-teal"></i> ${l.doctor_name}</strong>
                <span class="badge badge-success">${l.status}</span>
            </div>
            <div class="small-text text-muted">${l.qualification} | ${l.timestamp}</div>
            <p style="margin-top:0.2rem; font-size:0.82rem; color:var(--text-main);">${l.comments}</p>
        </div>
    `).join("");
}

// Patient Registration Modal
function openRegistrationModal() {
    document.getElementById("registerModal").classList.remove("hidden");
    if (typeof gsap !== "undefined") {
        gsap.from("#registerModal .modal-card", { duration: 0.4, scale: 0.9, opacity: 0, ease: "back.out(1.4)" });
    }
}

function closeRegistrationModal() {
    document.getElementById("registerModal").classList.add("hidden");
}

async function handleRegistrationSubmit(e) {
    e.preventDefault();
    const payload = {
        name: document.getElementById("regName").value,
        age: parseInt(document.getElementById("regAge").value),
        gender: document.getElementById("regGender").value,
        contact: document.getElementById("regContact").value,
        primary_physician: document.getElementById("regPhysician").value,
        clinical_summary: document.getElementById("regSummary").value
    };

    try {
        const res = await fetch(`${API_BASE}/api/patients/register`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            const data = await res.json();
            const newP = data.patient;
            patientDataMap[newP.id] = newP;

            const select = document.getElementById("patientSelect");
            const opt = document.createElement("option");
            opt.value = newP.id;
            opt.innerText = `${newP.id} | ${newP.name} (${newP.age}${newP.gender[0]}) - New Intake`;
            opt.selected = true;
            select.insertBefore(opt, select.firstChild);

            activeMrn = newP.id;
            closeRegistrationModal();
            loadPatientData(newP.id);
        }
    } catch (err) {
        console.warn("Registration API error:", err);
    }
}

// Patients Retrieval
async function fetchPatients() {
    try {
        const res = await fetch(`${API_BASE}/api/patients`);
        if (res.ok) {
            const data = await res.json();
            data.patients.forEach(p => {
                patientDataMap[p.id] = p;
            });
            loadPatientData("MRN-2026-1023");
        }
    } catch (err) {
        console.warn("API unavailable:", err);
    }
}

async function loadPatientData(mrn) {
    const patient = patientDataMap[mrn];
    if (!patient) return;

    document.getElementById("dispPatientName").innerText = patient.name;
    document.getElementById("dispPatientMrn").innerText = patient.mrn || patient.id;
    document.getElementById("dispPatientAge").innerText = patient.age;
    document.getElementById("dispPatientGender").innerText = patient.gender;
    document.getElementById("dispPhysician").innerText = patient.primary_physician || "Dr. Rohan Shinde, MBBS, MD Psychiatry";
    document.getElementById("dispPatientNotes").innerText = patient.clinical_summary;

    const targetBadge = document.getElementById("voiceTargetPatientBadge");
    if (targetBadge) targetBadge.innerText = `${patient.mrn || patient.id} (${patient.name})`;
    const voiceMrnDisp = document.getElementById("voiceActiveMrnDisplay");
    if (voiceMrnDisp) voiceMrnDisp.innerText = patient.mrn || patient.id;

    populateIntakeForm(patient.features);
    await runInference(patient.features, patient.id, patient.name, patient.age, patient.gender);
}

function populateIntakeForm(f) {
    document.getElementById("inpPhq9").value = f.phq9_score || 0;
    document.getElementById("inpGad7").value = f.gad7_score || 0;
    document.getElementById("inpMdq").value = f.mdq_score || 0;
    document.getElementById("inpWho5").value = f.who5_wellbeing || 50;

    document.getElementById("inpSleep").value = f.sleep_hours || 7;
    document.getElementById("inpSocial").value = f.social_interaction_score || 50;
    document.getElementById("inpActivity").value = f.physical_activity_min || 30;
    document.getElementById("inpStress").value = f.stress_level || 5;

    updateFormLabels();
}

function updateFormLabels() {
    document.getElementById("lblPhq9").innerText = document.getElementById("inpPhq9").value;
    document.getElementById("lblGad7").innerText = document.getElementById("inpGad7").value;
    document.getElementById("lblMdq").innerText = document.getElementById("inpMdq").value;
    document.getElementById("lblWho5").innerText = document.getElementById("inpWho5").value;

    document.getElementById("lblSleep").innerText = document.getElementById("inpSleep").value;
    document.getElementById("lblSocial").innerText = document.getElementById("inpSocial").value;
    document.getElementById("lblActivity").innerText = document.getElementById("inpActivity").value;
    document.getElementById("lblStress").innerText = document.getElementById("inpStress").value;
}

// Inference & Doctor Report Fetch
async function runInference(features, id = "MRN-2026-1023", name = "Vedant", age = 21, gender = "Male") {
    const payload = {
        id: id,
        name: name,
        age: parseInt(age),
        gender: gender,
        phq9_score: parseFloat(features.phq9_score || 0),
        gad7_score: parseFloat(features.gad7_score || 0),
        mdq_score: parseFloat(features.mdq_score || 0),
        who5_wellbeing: parseFloat(features.who5_wellbeing || 50),
        sleep_hours: parseFloat(features.sleep_hours || 7),
        social_interaction_score: parseFloat(features.social_interaction_score || 50),
        physical_activity_min: parseFloat(features.physical_activity_min || 30),
        stress_level: parseInt(features.stress_level || 5),
        speech_pitch_hz: parseFloat(features.speech_pitch_hz || 140),
        speaking_rate_wpm: parseFloat(features.speaking_rate_wpm || 130),
        pause_frequency_ppm: parseFloat(features.pause_frequency_ppm || 12),
        speech_tone_var: parseFloat(features.speech_tone_var || 30),
        sentiment_score: parseFloat(features.sentiment_score || 0),
        hopelessness_prob: parseFloat(features.hopelessness_prob || 0.1)
    };

    try {
        const [predRes, xaiRes, repRes] = await Promise.all([
            fetch(`${API_BASE}/api/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
            fetch(`${API_BASE}/api/explain`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
            fetch(`${API_BASE}/api/doctor-report`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
        ]);

        if (predRes.ok && xaiRes.ok && repRes.ok) {
            currentPrediction = await predRes.json();
            currentXAI = await xaiRes.json();
            currentDoctorReport = await repRes.json();

            updateDashboardUI(currentPrediction, currentXAI, currentDoctorReport);
        }
    } catch (err) {
        console.warn("Inference API error:", err);
    }
}

function updateDashboardUI(pred, xai, report) {
    animateCounter("dispConfidence", pred.confidence_score, "%");
    document.getElementById("dispPatientStatus").innerText = pred.risk_categories.depression === "HIGH" ? "HIGH RISK ADVISORY" : (pred.risk_categories.depression === "MODERATE" ? "MODERATE RISK" : "STABLE");

    // Resolve patient name & id from multiple sources
    const patientDisplayName = report?.patient_name || pred.patient_name || (patientDataMap[activeMrn] && patientDataMap[activeMrn].name) || activeMrn;
    const patientDisplayId   = report?.patient_id   || pred.patient_id   || activeMrn;

    // Alert Banner
    const banner = document.getElementById("alertBanner");
    if (pred.alert_triggered) {
        banner.classList.remove("hidden");
        document.getElementById("alertTitle").innerText = `${pred.alert_level} CLINICAL ADVISORY`;
        document.getElementById("alertMessage").innerText = `Patient ${patientDisplayName} (${patientDisplayId}) presents with Depression (${pred.risk_scores.depression}%) and Suicide Risk (${pred.risk_scores.suicide}%). Immediate clinician evaluation recommended.`;
    } else {
        banner.classList.add("hidden");
    }

    // Risk Cards & Gauges
    updateGauge("gaugeDepression", "valDepression", "badgeDepression", pred.risk_scores.depression, pred.risk_categories.depression, "#F43F5E");
    updateGauge("gaugeAnxiety", "valAnxiety", "badgeAnxiety", pred.risk_scores.anxiety, pred.risk_categories.anxiety, "#F59E0B");
    updateGauge("gaugeBipolar", "valBipolar", "badgeBipolar", pred.risk_scores.bipolar, pred.risk_categories.bipolar, "#10B981");
    updateGauge("gaugeSuicide", "valSuicide", "badgeSuicide", pred.risk_scores.suicide, pred.risk_categories.suicide, "#8B5CF6");

    document.getElementById("valPhq9").innerText = `${document.getElementById("inpPhq9").value} / 27`;
    document.getElementById("valGad7").innerText = `${document.getElementById("inpGad7").value} / 21`;
    document.getElementById("valMdq").innerText = `${document.getElementById("inpMdq").value} / 13`;

    // Derived Chips
    document.getElementById("chipSleepRisk").innerText = pred.derived_indices.sleep_risk_index;
    document.getElementById("chipSocialIso").innerText = pred.derived_indices.social_isolation_index;
    document.getElementById("chipMoodVar").innerText = pred.derived_indices.mood_variability_index;
    document.getElementById("chipDepIndex").innerText = pred.derived_indices.depression_index;

    // Top Factors & Actions
    renderTopFactorsList(xai.top_contributing_factors);
    renderRecommendedActions(report.preventive_actions);

    // Trend & SHAP Charts
    renderTrendChart(pred.historical_trend);
    renderShapChart(xai.top_contributing_factors);
    renderGlobalShapList(xai.global_shap_importances);
    document.getElementById("xaiSummaryText").innerText = xai.explanation_summary;

    // Official Doctor Report UI & Audit Logs
    updateOfficialDoctorReportUI(report);
    renderConsultationAuditLogs(report.consultation_logs);
}

function updateOfficialDoctorReportUI(report) {
    document.getElementById("repMrn").innerText = report.patient_id;
    document.getElementById("repPatientName").innerText = report.patient_name;
    document.getElementById("repAgeGender").innerText = `${report.age} Y / ${report.gender}`;
    document.getElementById("repConfidence").innerText = `${currentPrediction.confidence_score}%`;
    document.getElementById("repStatus").innerText = report.risk_categories.depression === "HIGH" ? "HIGH RISK ADVISORY" : "STABLE";

    document.getElementById("repDepScore").innerText = `${report.risk_scores.depression}%`;
    document.getElementById("repDepCat").innerText = report.risk_categories.depression;

    document.getElementById("repAnxScore").innerText = `${report.risk_scores.anxiety}%`;
    document.getElementById("repAnxCat").innerText = report.risk_categories.anxiety;

    document.getElementById("repBipScore").innerText = `${report.risk_scores.bipolar}%`;
    document.getElementById("repBipCat").innerText = report.risk_categories.bipolar;

    document.getElementById("repSuiScore").innerText = `${report.risk_scores.suicide}%`;
    document.getElementById("repSuiCat").innerText = report.risk_categories.suicide;

    const xaiBox = document.getElementById("repXaiBreakdown");
    xaiBox.innerHTML = `<strong>Top Clinical Risk Drivers:</strong><ul style="margin-top:0.4rem; padding-left:1.2rem;">` +
        report.top_contributing_factors.slice(0, 4).map(f => `<li><strong>${f.display_name}</strong>: Contributes ${f.percentage}% to overall calculated risk.</li>`).join("") +
        `</ul>`;

    const actList = document.getElementById("repActionsList");
    actList.innerHTML = report.preventive_actions.map(a => `<li><strong>${a}</strong></li>`).join("");

    document.getElementById("repDoctorNote").innerText = report.doctor_note;

    const cert = report.doctor_certificate;
    document.getElementById("repDocName").innerText = cert.doctor_name;
    document.getElementById("repDocTitle").innerText = `${cert.doctor_name} - ${cert.designation}`;
    document.getElementById("repSigHash").innerText = cert.digital_signature_hash;
    document.getElementById("repSealBadge").innerText = cert.doctor_seal_badge;
}

// Document Upload & OCR Parser
function triggerFileInput() {
    document.getElementById("fileInput").click();
}

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(evt) {
        document.getElementById("inpOcrText").value = evt.target.result;
        runDocumentOCR();
    };
    reader.readAsText(file);
}

async function runDocumentOCR() {
    const text = document.getElementById("inpOcrText").value;
    if (!text.trim()) return;

    try {
        const res = await fetch(`${API_BASE}/api/scan-document`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ document_text: text })
        });

        if (res.ok) {
            const data = await res.json();
            const feats = data.extracted_features;
            
            if (feats.phq9_score !== undefined) document.getElementById("inpPhq9").value = feats.phq9_score;
            if (feats.gad7_score !== undefined) document.getElementById("inpGad7").value = feats.gad7_score;
            if (feats.mdq_score !== undefined) document.getElementById("inpMdq").value = feats.mdq_score;
            if (feats.sleep_hours !== undefined) document.getElementById("inpSleep").value = feats.sleep_hours;
            if (feats.stress_level !== undefined) document.getElementById("inpStress").value = feats.stress_level;
            
            updateFormLabels();
            alert(`Document scanned! Auto-filled extracted parameters: ${data.fields_found.join(", ")}`);
        }
    } catch (err) {
        console.warn("OCR API Error:", err);
    }
}

// Live Microphone Web Audio API Recording & Feature Extraction
async function toggleMicrophoneRecording() {
    const btn = document.getElementById("btnMicRecord");
    const txt = document.getElementById("txtMicState");
    const statusTxt = document.getElementById("txtMicStatus");
    const wave = document.getElementById("waveformBox");

    if (!isRecordingMic) {
        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            analyserNode = audioCtx.createAnalyser();
            analyserNode.fftSize = 2048;

            const source = audioCtx.createMediaStreamSource(micStream);
            source.connect(analyserNode);

            isRecordingMic = true;
            btn.classList.add("recording");
            wave.classList.add("active");
            txt.innerText = "Stop Live Recording";
            statusTxt.innerText = "Live Microphone Stream Active";

            processLiveAudio();
        } catch (err) {
            console.warn("Microphone access error:", err);
            statusTxt.innerText = "Microphone Permission Denied or Unavailable";
        }
    } else {
        if (micStream) micStream.getTracks().forEach(track => track.stop());
        if (audioCtx) audioCtx.close();
        if (audioAnimFrame) cancelAnimationFrame(audioAnimFrame);

        isRecordingMic = false;
        btn.classList.remove("recording");
        wave.classList.remove("active");
        txt.innerText = "Start Microphone Live Recording";
        statusTxt.innerText = "Recording Processed & Acoustic Features Extracted";
    }
}

function processLiveAudio() {
    if (!isRecordingMic || !analyserNode) return;

    const bufferLength = analyserNode.fftSize;
    const dataArray = new Float32Array(bufferLength);
    analyserNode.getFloatTimeDomainData(dataArray);

    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sum / bufferLength);
    const db = Math.max(-60, Math.round(20 * Math.log10(rms + 1e-6)));

    const pitch = autoCorrelatePitch(dataArray, audioCtx.sampleRate);

    document.getElementById("liveEnergyDb").innerText = `${db} dB`;
    document.getElementById("livePitchHz").innerText = pitch > 0 ? `${Math.round(pitch)} Hz` : "-- Hz";
    document.getElementById("livePauseCount").innerText = rms < 0.02 ? "Pause Detected" : "Speaking";

    audioAnimFrame = requestAnimationFrame(processLiveAudio);
}

function autoCorrelatePitch(buf, sampleRate) {
    let SIZE = buf.length;
    let rms = 0;
    for (let i = 0; i < SIZE; i++) {
        let val = buf[i];
        rms += val * val;
    }
    rms = Math.sqrt(rms / SIZE);
    if (rms < 0.01) return -1;

    let r1 = 0, r2 = SIZE - 1, thres = 0.2;
    for (let i = 0; i < SIZE / 2; i++) {
        if (Math.abs(buf[i]) < thres) { r1 = i; break; }
    }
    for (let i = 1; i < SIZE / 2; i++) {
        if (Math.abs(buf[SIZE - i]) < thres) { r2 = SIZE - i; break; }


    }

    buf = buf.slice(r1, r2);
    SIZE = buf.length;

    let c = new Array(SIZE).fill(0);
    for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE - i; j++) {
            c[i] = c[i] + buf[j] * buf[j + i];
        }
    }

    let d = 0; while (c[d] > c[d + 1]) d++;
    let maxval = -1, maxpos = -1;
    for (let i = d; i < SIZE; i++) {
        if (c[i] > maxval) { maxval = c[i]; maxpos = i; }
    }
    let T0 = maxpos;
    return sampleRate / T0;
}

// Helpers & Chart Renderers
function renderTopFactorsList(factors) {
    const container = document.getElementById("factorsListContainer");
    container.innerHTML = "";
    factors.slice(0, 5).forEach((f, idx) => {
        const div = document.createElement("div");
        div.className = "factor-item";
        div.innerHTML = `
            <div class="factor-row">
                <span>${idx + 1}. ${f.display_name}</span>
                <strong>${f.percentage}%</strong>
            </div>
            <div class="progress-bar"><div class="progress-fill" style="width: ${Math.min(f.percentage * 3, 100)}%; background: var(--color-teal);"></div></div>
        `;
        container.appendChild(div);
    });
}

function renderRecommendedActions(actions) {
    const list = document.getElementById("recommendedActionsList");
    list.innerHTML = "";
    actions.forEach(a => {
        const li = document.createElement("li");
        li.innerHTML = `<i class="fa-solid fa-circle-check text-emerald"></i> <strong>${a}</strong>`;
        list.appendChild(li);
    });
}

function initGaugeCharts() {
    ["gaugeDepression", "gaugeAnxiety", "gaugeBipolar", "gaugeSuicide"].forEach(id => {
        const ctx = document.getElementById(id).getContext("2d");
        gauges[id] = new Chart(ctx, {
            type: "doughnut",
            data: { datasets: [{ data: [50, 50], backgroundColor: ["#06B6D4", "rgba(255,255,255,0.06)"], borderWidth: 0 }] },
            options: { cutout: "78%", responsive: true, maintainAspectRatio: false, plugins: { tooltip: { enabled: false } } }
        });
    });
}

function updateGauge(gaugeId, valId, badgeId, value, category, color) {
    if (gauges[gaugeId]) {
        gauges[gaugeId].data.datasets[0].backgroundColor = [color, "rgba(255,255,255,0.06)"];
        gauges[gaugeId].data.datasets[0].data = [value, Math.max(0, 100 - value)];
        gauges[gaugeId].update();
    }
    animateCounter(valId, value, "%");
    const b = document.getElementById(badgeId);
    b.innerText = `${category} (${value}%)`;
    b.className = `badge ${category === "HIGH" ? "badge-danger" : (category === "MODERATE" ? "badge-warning" : "badge-success")}`;
}

function initTrendChart() {
    const ctx = document.getElementById("trendChart").getContext("2d");
    chartTrend = new Chart(ctx, {
        type: "line",
        data: {
            labels: ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11", "W12"],
            datasets: [
                { label: "Depression Risk", data: Array(12).fill(50), borderColor: "#F43F5E", tension: 0.3 },
                { label: "Anxiety Risk", data: Array(12).fill(40), borderColor: "#F59E0B", tension: 0.3 },
                { label: "Bipolar Risk", data: Array(12).fill(20), borderColor: "#10B981", tension: 0.3 },
                { label: "Suicide Risk Alert", data: Array(12).fill(30), borderColor: "#8B5CF6", borderDash: [5, 5], tension: 0.3 }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 100 } } }
    });
}

function renderTrendChart(trend) {
    if (!chartTrend || !trend) return;
    chartTrend.data.labels = trend.map(t => t.week);
    chartTrend.data.datasets[0].data = trend.map(t => t.depression);
    chartTrend.data.datasets[1].data = trend.map(t => t.anxiety);
    chartTrend.data.datasets[2].data = trend.map(t => t.bipolar);
    chartTrend.data.datasets[3].data = trend.map(t => t.suicide);
    chartTrend.update();
}

function initShapChart() {
    const ctx = document.getElementById("shapChart").getContext("2d");
    chartShap = new Chart(ctx, {
        type: "bar",
        data: { labels: [], datasets: [{ data: [], backgroundColor: "#06B6D4" }] },
        options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });
}

function renderShapChart(factors) {
    if (!chartShap || !factors) return;
    chartShap.data.labels = factors.map(f => f.display_name);
    chartShap.data.datasets[0].data = factors.map(f => f.percentage);
    chartShap.update();
}

function renderGlobalShapList(importances) {
    const container = document.getElementById("globalShapList");
    if (!container || !importances) return;
    container.innerHTML = "";
    importances.forEach((imp, i) => {
        const div = document.createElement("div");
        div.className = "factor-item";
        div.innerHTML = `
            <div class="factor-row"><span>${i + 1}. ${imp.name}</span><strong>${(imp.importance * 100).toFixed(0)}%</strong></div>
            <div class="progress-bar"><div class="progress-fill" style="width:${imp.importance * 100}%; background:var(--color-teal);"></div></div>
        `;
        container.appendChild(div);
    });
}

async function handleIntakeSubmit(e) {
    e.preventDefault();
    const features = {
        phq9_score: document.getElementById("inpPhq9").value,
        gad7_score: document.getElementById("inpGad7").value,
        mdq_score: document.getElementById("inpMdq").value,
        who5_wellbeing: document.getElementById("inpWho5").value,
        sleep_hours: document.getElementById("inpSleep").value,
        social_interaction_score: document.getElementById("inpSocial").value,
        physical_activity_min: document.getElementById("inpActivity").value,
        stress_level: document.getElementById("inpStress").value
    };

    const patient = patientDataMap[activeMrn] || { id: activeMrn, name: "Intake Patient", age: 25, gender: "Male" };
    await runInference(features, patient.id, patient.name, patient.age, patient.gender);
    switchTab("overview");
}

async function fetchModelMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/metrics`);
        if (res.ok) {
            const data = await res.json();
            const ens = data.metrics.final_ensemble;
            document.getElementById("statAcc").innerText = `${(ens.accuracy * 100).toFixed(0)}%`;
            document.getElementById("statSens").innerText = `${(ens.sensitivity * 100).toFixed(0)}%`;
            document.getElementById("statSpec").innerText = `${(ens.specificity * 100).toFixed(0)}%`;
            document.getElementById("statAuc").innerText = ens.roc_auc.toFixed(3);
            document.getElementById("statF1").innerText = ens.f1_score.toFixed(3);
            document.getElementById("statKappa").innerText = ens.kappa_score.toFixed(3);
        }
    } catch (err) {
        console.warn("Metrics error:", err);
    }
}

// ==========================================
// VOICE INTERVIEW ASSISTANT LOGIC
// ==========================================
let voiceQuestions = [
    { text: "Hi there. How have you been sleeping lately? About how many hours per night?", context: "sleep" },
    { text: "How would you describe your mood or emotions over the past few weeks?", context: "mood" },
    { text: "Are you getting any physical activity or exercise during the week?", context: "activity" },
    { text: "On a scale of 1 to 10, how stressed are you feeling right now?", context: "stress" },
    { text: "Have you ever had any past episodes of severe depression or anxiety before this?", context: "episodes" },
    { text: "Is there any history of mental health conditions in your immediate family?", context: "family" },
    { text: "Are you currently taking any prescription medications? If so, how many?", context: "medications" },
    { text: "Do you have any other chronic medical conditions like diabetes or asthma?", context: "conditions" }
];
let currentQuestionIndex = 0;
let voiceExtractedData = {};
let voiceTranscriptLog = [];

function speakText(text, callback) {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onend = callback;
    synth.speak(utterance);
}

function startVoiceInterview() {
    document.getElementById('startVoiceBtn').style.display = 'none';
    currentQuestionIndex = 0;
    voiceExtractedData = {};
    voiceTranscriptLog = [];
    const banner = document.getElementById('voiceCompletionBanner');
    if (banner) banner.style.display = 'none';
    askNextVoiceQuestion();
}

function askNextVoiceQuestion() {
    if (currentQuestionIndex >= voiceQuestions.length) {
        document.getElementById('voiceQuestionText').innerText = "Interview complete. Thank you.";
        document.getElementById('voiceTranscriptText').innerText = "All responses processed.";
        
        // Show prominent completion banner
        const banner = document.getElementById('voiceCompletionBanner');
        if (banner) {
            banner.style.display = 'block';
            if (typeof gsap !== "undefined") {
                gsap.from(banner, { duration: 0.4, y: 15, opacity: 0, ease: "power2.out" });
            }
        }
        const voiceMrnDisp = document.getElementById('voiceActiveMrnDisplay');
        if (voiceMrnDisp) voiceMrnDisp.innerText = activeMrn;

        const btn = document.getElementById('voiceTransferBtn');
        if (btn) btn.disabled = false;
        return;
    }

    const q = voiceQuestions[currentQuestionIndex];
    document.getElementById('voiceQuestionText').innerText = q.text;
    document.getElementById('voiceTranscriptText').innerText = "(AI Speaking...)";
    
    speakText(q.text, () => {
        startListening(q.context);
    });
}

let voiceFollowupAttempts = 0;

async function handleVoiceTranscript(transcript, context) {
    document.getElementById('voiceTranscriptText').innerText = transcript;
    document.getElementById('voiceStatusIndicator').style.display = 'none';
    
    // Log transcript
    voiceTranscriptLog.push(`[${context.toUpperCase()}]: "${transcript}"`);

    // Clear simulation input if used
    const simInp = document.getElementById('voiceSimInput');
    if (simInp) simInp.value = '';

    // Send to backend with question context
    try {
        const currentQ = voiceQuestions[currentQuestionIndex] ? voiceQuestions[currentQuestionIndex].text : "";
        const res = await fetch(`${API_BASE}/api/voice-intake`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ transcript, context, question_text: currentQ })
        });
        const data = await res.json();
        
        // Merge extracted features
        Object.assign(voiceExtractedData, data.extracted_features);
        
        // Update UI
        if (voiceExtractedData.sleep_hours) document.getElementById('voiceValSleep').innerText = voiceExtractedData.sleep_hours + " hrs";
        if (voiceExtractedData.physical_activity_min !== undefined) document.getElementById('voiceValActivity').innerText = voiceExtractedData.physical_activity_min + " mins";
        if (voiceExtractedData.stress_level) document.getElementById('voiceValStress').innerText = voiceExtractedData.stress_level + " / 10";
        if (voiceExtractedData.sentiment_score !== undefined) document.getElementById('voiceValSentiment').innerText = voiceExtractedData.sentiment_score.toFixed(2);
        
        // Display and speak AI Conversational Reflection / Pivot
        if (data.conversational_reply) {
            const box = document.getElementById('voiceAiFeedbackBox');
            const txt = document.getElementById('voiceAiFeedbackText');
            if (box && txt) {
                txt.innerText = data.conversational_reply;
                box.style.display = 'block';
            }
        }

        const replyToSpeak = data.conversational_reply || "Response noted.";
        
        // If abrupt/off-topic response detected, give patient an empathetic retry for this specific question
        if (data.requires_followup && voiceFollowupAttempts < 1) {
            voiceFollowupAttempts++;
            speakText(replyToSpeak, () => {
                setTimeout(() => {
                    const q = voiceQuestions[currentQuestionIndex];
                    if (q) {
                        document.getElementById('voiceTranscriptText').innerText = "(Listening for clarification...)";
                        startListening(q.context);
                    }
                }, 600);
            });
        } else {
            voiceFollowupAttempts = 0;
            speakText(replyToSpeak, () => {
                setTimeout(() => {
                    currentQuestionIndex++;
                    askNextVoiceQuestion();
                }, 800);
            });
        }
        
    } catch(e) {
        console.error("Voice parse error:", e);
        document.getElementById('voiceStatusIndicator').style.display = 'none';
        setTimeout(() => {
            currentQuestionIndex++;
            askNextVoiceQuestion();
        }, 1000);
    }
}

function submitManualVoiceText() {
    const simInp = document.getElementById('voiceSimInput');
    if (!simInp) return;
    const text = simInp.value.trim();
    if (!text) return;
    const q = voiceQuestions[currentQuestionIndex] || { context: "sleep" };
    handleVoiceTranscript(text, q.context);
}

function startListening(context) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        document.getElementById('voiceTranscriptText').innerText = "Speech API not available. You can type answers in the test box below!";
        return;
    }
    
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    
    document.getElementById('voiceStatusIndicator').style.display = 'block';
    document.getElementById('voiceTranscriptText').innerText = "Listening...";
    
    recognition.onresult = async (event) => {
        const transcript = event.results[0][0].transcript;
        handleVoiceTranscript(transcript, context);
    };
    
    recognition.onerror = (e) => {
        console.error(e);
        document.getElementById('voiceStatusIndicator').style.display = 'none';
        document.getElementById('voiceTranscriptText').innerText = "Microphone paused. You can speak again or type in the test box below.";
    };
    
    try {
        recognition.start();
    } catch (err) {
        console.warn("Recognition already active or blocked:", err);
    }
}

function resetVoiceInterview() {
    currentQuestionIndex = 0;
    voiceExtractedData = {};
    voiceTranscriptLog = [];
    voiceFollowupAttempts = 0;
    document.getElementById('voiceValSleep').innerText = "--";
    document.getElementById('voiceValActivity').innerText = "--";
    document.getElementById('voiceValStress').innerText = "--";
    document.getElementById('voiceValSentiment').innerText = "--";
    document.getElementById('voiceTranscriptText').innerText = "Listening...";
    const box = document.getElementById('voiceAiFeedbackBox');
    if (box) box.style.display = 'none';
    const banner = document.getElementById('voiceCompletionBanner');
    if (banner) banner.style.display = 'none';
    const startBtn = document.getElementById('startVoiceBtn');
    if (startBtn) startBtn.style.display = 'inline-flex';
    document.getElementById('voiceQuestionText').innerText = "Click 'Start Live Interview' to begin...";
}

async function generateReportFromVoiceData() {
    const btn = document.getElementById("voiceTransferBtn");
    const originalBtnHtml = btn ? btn.innerHTML : "";

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving to EHR & Compiling Report...';
    }

    try {
        const patient = patientDataMap[activeMrn] || { id: activeMrn, name: "Vedant", age: 21, gender: "Male" };
        
        // 1. Post to backend to save voice features into active patient's EHR and generate full doctor report
        const res = await fetch(`${API_BASE}/api/patients/${activeMrn}/save-voice-intake`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                patient_id: activeMrn,
                extracted_features: voiceExtractedData,
                transcripts: voiceTranscriptLog,
                clinical_notes: `Clinical Voice Assistant Intake completed for ${patient.name} (${activeMrn}). Sleep: ${voiceExtractedData.sleep_hours || '--'}h, Activity: ${voiceExtractedData.physical_activity_min || '--'}m, Stress: ${voiceExtractedData.stress_level || '--'}/10.`
            })
        });

        if (res.ok) {
            const data = await res.json();
            
            // 2. Update global state with the freshly generated prediction, XAI, and doctor report
            currentPrediction = data.prediction;
            currentXAI = data.xai;
            currentDoctorReport = data.doctor_report;

            // 3. Update active patient in patientDataMap
            if (patientDataMap[activeMrn]) {
                patientDataMap[activeMrn].features = data.features;
                if (data.doctor_report && data.doctor_report.doctor_note) {
                    patientDataMap[activeMrn].clinical_summary = data.doctor_report.doctor_note;
                }
            }

            // 4. Update UI across dashboard and official doctor report
            updateDashboardUI(currentPrediction, currentXAI, currentDoctorReport);

            // 5. Update Voice Banner in Doctor Report tab
            const repBanner = document.getElementById("repVoiceSuccessBanner");
            const repText = document.getElementById("repVoiceSuccessText");
            if (repBanner && repText) {
                repText.innerText = `Data securely committed to Patient EHR (${activeMrn} - ${patient.name}). Extracted Sleep: ${voiceExtractedData.sleep_hours || 7}h, Stress: ${voiceExtractedData.stress_level || 5}/10, Sentiment: ${(voiceExtractedData.sentiment_score || 0).toFixed(2)}. Doctor diagnostic SOAP note & cryptographic certificate compiled.`;
                repBanner.classList.remove("hidden");
            }

            // 6. Direct user smoothly to the Official Doctor Diagnostic Report tab
            switchTab('report');
            window.scrollTo({ top: 0, behavior: 'smooth' });

        } else {
            console.warn("Backend save voice intake returned non-200. Falling back to local inference.");
            transferVoiceDataToFormFallback();
        }
    } catch (err) {
        console.error("Error generating report from voice data:", err);
        transferVoiceDataToFormFallback();
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalBtnHtml || '<i class="fa-solid fa-file-medical"></i> Generate Doctor Report from Voice';
        }
    }
}

function transferVoiceDataToFormFallback() {
    // Fallback if backend route had a network glitch
    if (voiceExtractedData.sleep_hours) {
        document.getElementById('inSleep').value = voiceExtractedData.sleep_hours;
        quizData.sleep_hours = voiceExtractedData.sleep_hours;
    }
    const patient = patientDataMap[activeMrn] || { id: activeMrn, name: "Intake Patient", age: 25, gender: "Male" };
    runInference(patient.features, patient.id, patient.name, patient.age, patient.gender).then(() => {
        switchTab('report');
    });
}

// Keep transferVoiceDataToForm as alias
const transferVoiceDataToForm = generateReportFromVoiceData;

// ==========================================
// GEMINI AI CONFIGURATION & MODAL HANDLERS
// ==========================================
async function fetchGeminiStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/config/gemini-status`);
        if (res.ok) {
            const data = await res.json();
            const btn = document.getElementById("geminiStatusBtn");
            const txt = document.getElementById("geminiStatusText");
            if (!btn || !txt) return;
            if (data.configured) {
                btn.classList.add("connected");
                txt.innerHTML = '<i class="fa-solid fa-circle-check text-emerald"></i> Gemini Active';
                btn.setAttribute("title", `Google Gemini Active (${data.model})`);
            } else {
                btn.classList.remove("connected");
                txt.innerHTML = '<i class="fa-solid fa-server text-teal"></i> Local Mode';
                btn.setAttribute("title", "Running on Local Explainable Heuristic Engine (Offline Safe)");
            }
        }
    } catch (e) {
        console.warn("Gemini status check error:", e);
    }
}

function openGeminiModal() {
    const modal = document.getElementById("geminiModal");
    if (!modal) return;
    modal.classList.remove("hidden");
    const feedback = document.getElementById("geminiStatusFeedback");
    if (feedback) feedback.innerHTML = "";
    if (typeof gsap !== "undefined") {
        gsap.from("#geminiModal .modal-card", { duration: 0.35, scale: 0.92, opacity: 0, ease: "back.out(1.4)" });
    }
}

function closeGeminiModal() {
    const modal = document.getElementById("geminiModal");
    if (modal) modal.classList.add("hidden");
}

async function saveGeminiKey() {
    const keyInp = document.getElementById("inpGeminiKey");
    const key = keyInp ? keyInp.value.trim() : "";
    const feedback = document.getElementById("geminiStatusFeedback");
    if (!key) {
        if (feedback) feedback.innerHTML = '<span class="text-rose"><i class="fa-solid fa-triangle-exclamation"></i> Please paste a valid Gemini API key.</span>';
        return;
    }
    
    if (feedback) feedback.innerHTML = '<span class="text-teal"><i class="fa-solid fa-spinner fa-spin"></i> Validating key with Google Gemini API...</span>';

    try {
        const res = await fetch(`${API_BASE}/api/config/gemini-key`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: key })
        });
        const data = await res.json();
        if (data.status === "CONNECTED") {
            if (feedback) feedback.innerHTML = `<span class="text-emerald"><i class="fa-solid fa-circle-check"></i> Connected! Model: ${data.model}</span>`;
            fetchGeminiStatus();
            setTimeout(() => {
                closeGeminiModal();
            }, 1200);
        } else {
            if (feedback) feedback.innerHTML = `<span class="text-rose"><i class="fa-solid fa-circle-xmark"></i> ${data.message || 'Validation failed.'}</span>`;
        }
    } catch (e) {
        if (feedback) feedback.innerHTML = `<span class="text-rose"><i class="fa-solid fa-circle-xmark"></i> Network error: ${e.message}</span>`;
    }
}

async function clearGeminiKey() {
    const feedback = document.getElementById("geminiStatusFeedback");
    try {
        await fetch(`${API_BASE}/api/config/gemini-key`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: "" })
        });
        const keyInp = document.getElementById("inpGeminiKey");
        if (keyInp) keyInp.value = "";
        if (feedback) feedback.innerHTML = '<span class="text-teal"><i class="fa-solid fa-info-circle"></i> Switched to 100% offline Local Heuristic Mode.</span>';
        fetchGeminiStatus();
        setTimeout(() => {
            closeGeminiModal();
        }, 1200);
    } catch (e) {
        if (feedback) feedback.innerHTML = `<span class="text-rose">Error resetting: ${e.message}</span>`;
    }
}
