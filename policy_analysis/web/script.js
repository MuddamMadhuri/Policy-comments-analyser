document.addEventListener('DOMContentLoaded', () => {
    const commentInput = document.getElementById('commentInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    const loadingSection = document.getElementById('loading');
    const errorSection = document.getElementById('error');

    const sentimentResult = document.getElementById('sentimentResult');
    const stanceResult = document.getElementById('stanceResult');
    const topicResult = document.getElementById('topicResult');
    const explanationText = document.getElementById('explanationText');

    const API_URL = '/predict';

    // Chart Instances
    let sentimentChart = null;
    let trendChart = null;
    let emotionChart = null;

    analyzeBtn.addEventListener('click', async () => {
        const text = commentInput.value.trim();
        const policy = document.getElementById('policySelect').value;

        if (!text) {
            alert('Please enter a comment.');
            return;
        }

        // Reset UI
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text, policy }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            // --- Logic for Display ---
            let displaySentiment = data.sentiment;

            // Insight Logic
            let insight = "";
            if (data.sentiment === "Positive") {
                insight = "This comment shows strong alignment with the policy goals. Key drivers appear to be economic benefits and community development.";
            } else if (data.sentiment === "Negative") {
                insight = "Negative sentiment spikes around implementation timeline clauses and potential environmental impact concerns.";
            } else if (data.sentiment === "Mixed") {
                insight = "This comment reflects conflicting viewpoints, acknowledging both benefits and significant concerns or conditions.";
            } else if (data.sentiment === "Invalid") {
                displaySentiment = "Invalid Input";
                insight = "The input does not contain enough meaningful text to analyze. Please provide a more detailed comment.";
            } else {
                insight = "The sentiment is neutral, suggesting a query or a balanced view without strong emotional language.";
            }

            if (data.sentiment === "Neutral" && data.stance === "Support") {
                displaySentiment = "Mildly Positive";
                insight = "This comment supports the policy but expresses low emotional intensity, typical of formal or conditional approval.";
            }

            // Update Text Results
            sentimentResult.textContent = displaySentiment;
            stanceResult.textContent = data.stance;

            const policySelect = document.getElementById('policySelect');
            const selectedPolicyName = policySelect.options[policySelect.selectedIndex].text;
            topicResult.textContent = selectedPolicyName;

            explanationText.textContent = insight;

            // Style Sentiment Color
            sentimentResult.className = 'result-value';
            const lowerSent = displaySentiment.toLowerCase();
            if (lowerSent.includes('positive') || lowerSent.includes('support')) {
                sentimentResult.style.color = '#16a34a';
            } else if (lowerSent.includes('negative') || lowerSent.includes('oppose')) {
                sentimentResult.style.color = '#dc2626';
            } else if (lowerSent.includes('mixed')) {
                sentimentResult.style.color = '#8b5cf6'; // Purple
            } else if (lowerSent.includes('invalid')) {
                sentimentResult.style.color = '#9ca3af'; // Gray
            } else {
                sentimentResult.style.color = '#ca8a04';
            }

            // --- Render Charts ---
            renderCharts(displaySentiment, data.stance);

            loadingSection.classList.add('hidden');
            resultsSection.classList.remove('hidden');

        } catch (error) {
            console.error('Error:', error);
            loadingSection.classList.add('hidden');
            errorSection.classList.remove('hidden');
        }
    });

    function renderCharts(sentiment, stance) {
        // 1. Sentiment Donut Chart
        const ctxSent = document.getElementById('sentimentChart').getContext('2d');
        if (sentimentChart) sentimentChart.destroy();

        // Simulate values based on sentiment
        let sentData = [10, 10, 80]; // Pos, Neu, Neg example
        if (sentiment.includes("Positive")) sentData = [85, 10, 5];
        else if (sentiment.includes("Negative")) sentData = [5, 10, 85];
        else if (sentiment.includes("Mixed")) sentData = [40, 20, 40];
        else if (sentiment.includes("Invalid")) sentData = [0, 100, 0]; // All Neutral/Gray
        else sentData = [20, 60, 20];

        sentimentChart = new Chart(ctxSent, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: sentData,
                    backgroundColor: sentiment.includes("Invalid") ? ['#e5e7eb', '#9ca3af', '#e5e7eb'] : ['#16a34a', '#ca8a04', '#dc2626'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } }
            }
        });

        // 2. Trend Over Time (Mock)
        const ctxTrend = document.getElementById('trendChart').getContext('2d');
        if (trendChart) trendChart.destroy();

        // Mock Trend Data
        const labels = ['Week 1', 'Week 2', 'Draft Release', 'Week 4', 'Week 5'];
        let trendData = [0.2, 0.3, 0.1, -0.4, -0.2]; // Default
        if (sentiment.includes("Positive")) trendData = [0.1, 0.2, 0.6, 0.7, 0.8];
        if (sentiment.includes("Negative")) trendData = [0.0, -0.1, -0.5, -0.7, -0.8];
        if (sentiment.includes("Mixed")) trendData = [0.2, -0.3, 0.4, -0.2, 0.1];
        if (sentiment.includes("Invalid")) trendData = [0, 0, 0, 0, 0]; // Flat line

        trendChart = new Chart(ctxTrend, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Sentiment Score',
                    data: trendData,
                    borderColor: '#6A1B9A',
                    backgroundColor: 'rgba(106, 27, 154, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: false, min: -1, max: 1 } }
            }
        });

        // 3. Emotion Breakdown (Bar)
        const ctxEmo = document.getElementById('emotionChart').getContext('2d');
        if (emotionChart) emotionChart.destroy();

        // Mock Emotion Data based on sentiment
        let emoData = [10, 10, 10, 10, 10]; // Trust, Anger, Fear, Hope, Confusion
        if (sentiment.includes("Positive")) emoData = [80, 5, 5, 70, 10]; // High Trust, Hope
        if (sentiment.includes("Negative")) emoData = [10, 75, 40, 5, 30]; // High Anger, Fear
        if (sentiment.includes("Mixed")) emoData = [40, 30, 30, 40, 60]; // High Confusion, Balanced others
        if (sentiment.includes("Invalid")) emoData = [0, 0, 0, 0, 0]; // No emotion

        emotionChart = new Chart(ctxEmo, {
            type: 'bar',
            data: {
                labels: ['Trust', 'Anger', 'Fear', 'Hope', 'Confusion'],
                datasets: [{
                    label: 'Intensity',
                    data: emoData,
                    backgroundColor: [
                        '#3b82f6', // Trust (Blue)
                        '#ef4444', // Anger (Red)
                        '#a855f7', // Fear (Purple)
                        '#10b981', // Hope (Green)
                        '#f59e0b'  // Confusion (Orange)
                    ],
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, max: 100 } },
                plugins: { legend: { display: false } }
            }
        });
    }
});
