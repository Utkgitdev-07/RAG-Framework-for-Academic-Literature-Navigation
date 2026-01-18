/**
 * Frontend JavaScript for Enhanced RAG Literature Navigation System
 * 
 * This module handles all client-side functionality including:
 * - API communication with Flask backend
 * - User interface interactions
 * - Search result display and formatting
 * - Clustering visualization
 * - Multi-modal search indicators
 * 
 * Author: Research Team
 * Date: 2024
 */

// API Configuration - Base URL for backend REST API
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements - References to HTML elements for manipulation
const indexBtn = document.getElementById('indexBtn');
const statsBtn = document.getElementById('statsBtn');
const resetBtn = document.getElementById('resetBtn');
const searchBtn = document.getElementById('searchBtn');
const searchInput = document.getElementById('searchInput');
const useClustering = document.getElementById('useClustering');
const useHybrid = document.getElementById('useHybrid');
const topKSelect = document.getElementById('topKSelect');
const statusMessage = document.getElementById('statusMessage');
const statsPanel = document.getElementById('statsPanel');
const statsContent = document.getElementById('statsContent');
const resultsSection = document.getElementById('resultsSection');
const resultsTitle = document.getElementById('resultsTitle');
const resultsCount = document.getElementById('resultsCount');
const clustersView = document.getElementById('clustersView');
const listView = document.getElementById('listView');
const loadingIndicator = document.getElementById('loadingIndicator');

// Event Listeners
indexBtn.addEventListener('click', indexPapers);
statsBtn.addEventListener('click', showStats);
resetBtn.addEventListener('click', resetIndex);
searchBtn.addEventListener('click', performSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

/**
 * Generic API call function for communicating with Flask backend
 * 
 * Handles HTTP requests to the REST API, including error handling and
 * response parsing. All API calls go through this centralized function
 * to ensure consistent error handling and request formatting.
 * 
 * @param {string} endpoint - API endpoint path (e.g., '/search', '/index')
 * @param {string} method - HTTP method ('GET', 'POST', etc.)
 * @param {Object|null} data - Request body data (null for GET requests)
 * @returns {Promise<Object>} Parsed JSON response from server
 */
async function apiCall(endpoint, method = 'GET', data = null) {
    // Configure HTTP request options
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json',  // JSON content type for all requests
        },
    };

    // Add request body for POST/PUT requests
    if (data) {
        options.body = JSON.stringify(data);
    }

    try {
        // Execute HTTP request using Fetch API
        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        const result = await response.json();  // Parse JSON response
        return result;
    } catch (error) {
        // Handle network errors or parsing failures
        console.error('API Error:', error);
        showStatus('Error: ' + error.message, 'error');
        return { success: false, error: error.message };
    }
}

// UI Functions
function showStatus(message, type = 'success') {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
    setTimeout(() => {
        statusMessage.className = 'status-message';
    }, 5000);
}

function showLoading(show = true) {
    loadingIndicator.classList.toggle('hidden', !show);
}

// Index Papers
async function indexPapers() {
    showLoading(true);
    indexBtn.disabled = true;
    
    try {
        const result = await apiCall('/index', 'POST', {});
        
        if (result.success) {
            showStatus(`Successfully indexed ${result.num_documents} papers!`, 'success');
        } else {
            showStatus(`Error: ${result.error}`, 'error');
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
        indexBtn.disabled = false;
    }
}

// Show Stats
async function showStats() {
    showLoading(true);
    
    try {
        const result = await apiCall('/stats', 'GET');
        
        if (result.success) {
            const stats = result.index_stats;
            statsContent.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; color: #E6E6E6;">
                    <div style="color: #A1A1A1;">
                        <strong style="color: #E6E6E6; display: block; margin-bottom: 4px;">Indexed:</strong> ${stats.indexed ? '<span style="color: #00E676;">Yes</span>' : '<span style="color: #A1A1A1;">No</span>'}
                    </div>
                    <div style="color: #A1A1A1;">
                        <strong style="color: #E6E6E6; display: block; margin-bottom: 4px;">Documents:</strong> <span style="color: #00E676;">${stats.num_documents || 0}</span>
                    </div>
                    <div style="color: #A1A1A1;">
                        <strong style="color: #E6E6E6; display: block; margin-bottom: 4px;">PDF Files:</strong> <span style="color: #00E676;">${result.pdf_files_count || 0}</span>
                    </div>
                    <div style="color: #A1A1A1;">
                        <strong style="color: #E6E6E6; display: block; margin-bottom: 4px;">Embedding Dim:</strong> <span style="color: #A1A1A1;">${stats.embedding_dim || 'N/A'}</span>
                    </div>
                </div>
                <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid #1F252E; color: #A1A1A1;">
                    <strong style="color: #E6E6E6; display: block; margin-bottom: 4px;">Data Directory:</strong> <span style="font-size: 13px; word-break: break-all;">${result.data_directory}</span>
                </div>
            `;
            statsPanel.classList.remove('hidden');
        } else {
            showStatus(`Error: ${result.error}`, 'error');
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Reset Index
async function resetIndex() {
    if (!confirm('Are you sure you want to reset the index? This will delete all indexed papers.')) {
        return;
    }
    
    showLoading(true);
    
    try {
        const result = await apiCall('/reset', 'POST', {});
        
        if (result.success) {
            showStatus('Index reset successfully!', 'success');
            statsPanel.classList.add('hidden');
            resultsSection.classList.add('hidden');
        } else {
            showStatus(`Error: ${result.error}`, 'error');
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Execute search query using multi-modal retrieval system
 * 
 * This function implements the main search functionality by:
 * 1. Validating user input
 * 2. Collecting search parameters (query, top_k, clustering, hybrid mode)
 * 3. Sending request to backend API
 * 4. Displaying results with clustering and multi-modal indicators
 * 
 * Supports both text-only and hybrid (multi-modal) search modes.
 * Optionally applies topic clustering to organize results.
 */
async function performSearch() {
    // Extract and validate search query from input field
    const query = searchInput.value.trim();
    
    // Validate that query is not empty
    if (!query) {
        showStatus('Please enter a search query', 'error');
        return;
    }
    
    // Show loading indicator and disable search button to prevent duplicate requests
    showLoading(true);
    searchBtn.disabled = true;
    resultsSection.classList.add('hidden');  // Hide previous results
    
    try {
        // Collect search parameters from UI controls
        const topK = parseInt(topKSelect.value);          // Number of results to retrieve
        const useClust = useClustering.checked;          // Enable topic clustering
        const useHybridSearch = useHybrid.checked;       // Enable multi-modal search
        
        // Send search request to backend API
        const result = await apiCall('/search', 'POST', {
            query: query,                  // User's search query
            top_k: topK,                  // Maximum results to return
            use_clustering: useClust,     // Whether to cluster results
            use_hybrid: useHybridSearch   // Whether to use multi-modal search
        });
        
        // Handle successful response
        if (result.success) {
            displayResults(result);                      // Render search results
            resultsSection.classList.remove('hidden');   // Show results section
        } else {
            showStatus(`Error: ${result.error}`, 'error');
        }
    } catch (error) {
        // Handle unexpected errors
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        // Always restore UI state regardless of success/failure
        showLoading(false);
        searchBtn.disabled = false;
    }
}

/**
 * Display search results with appropriate visualization (clustered or list view)
 * 
 * This function renders search results based on whether clustering was applied.
 * It shows either:
 * - Clustered view: Results organized into topic groups with summaries (Improvement 2)
 * - List view: Flat ranked list of documents
 * 
 * Also displays multi-modal search indicators when hybrid search is active.
 * 
 * @param {Object} result - Search result object from API containing:
 *   - query: Original search query
 *   - num_results: Number of papers found
 *   - results: Array of ranked documents
 *   - clusters: Optional array of clustered results
 *   - use_hybrid: Whether multi-modal search was used
 *   - search_breakdown: Optional score breakdown for multi-modal search
 */
function displayResults(result) {
    // Update results header with query
    resultsTitle.textContent = `Search Results for: "${result.query}"`;
    
    // Display multi-modal search information if hybrid search was used (Improvement 1)
    if (result.use_hybrid) {
        displayMultimodalInfo(result);  // Show text vs metadata score breakdown
    }
    
    // Provide user feedback about result count and clustering availability
    if (result.num_results === 1) {
        // Warn user that clustering requires more papers
        resultsCount.innerHTML = `<span style="color: #A1A1A1;">⚠️ Only 1 paper indexed. Add more papers to see clustering and better search results!</span>`;
    } else {
        resultsCount.textContent = `${result.num_results} papers found`;
    }
    
    // Clear any previous results from the display
    clustersView.innerHTML = '';
    listView.innerHTML = '';
    
    // Determine which view to display: clusters or flat list
    if (result.clusters && result.clusters.length > 0 && result.num_results >= 3) {
        // Display clustered view (Improvement 2 visualization)
        displayClusters(result.clusters);                    // Render topic clusters
        clustersView.style.display = 'block';                // Show cluster container
        listView.style.display = 'none';                     // Hide list container
        document.getElementById('comparisonSection').classList.add('hidden');
    } else {
        // Display flat list view (traditional ranked results)
        displayList(result.results);                         // Render document list
        clustersView.style.display = 'none';                 // Hide cluster container
        listView.style.display = 'block';                    // Show list container
        document.getElementById('comparisonSection').classList.add('hidden');
        
        // Show informative message if clustering was requested but not available
        if (result.num_results < 3 && useClustering.checked) {
            listView.innerHTML = `
                <div class="warning-box">
                    <strong>⚠️ Clustering requires at least 3 papers.</strong>
                    <p>Currently indexed: ${result.num_results} paper(s). Please add more PDF papers to enable topic clustering (Improvement 2).</p>
                </div>
            ` + listView.innerHTML;
        }
    }
}

// Display Clusters
function displayClusters(clusters) {
    clusters.forEach((cluster, index) => {
        const clusterDiv = document.createElement('div');
        clusterDiv.className = 'cluster';
        
        clusterDiv.innerHTML = `
            <div class="cluster-header">
                <div class="cluster-title">Topic ${index + 1}</div>
                <span class="cluster-badge">${cluster.num_documents} papers</span>
            </div>
            <div class="cluster-summary">${cluster.summary}</div>
            <div class="cluster-keywords">
                ${cluster.keywords.map(kw => `<span class="keyword-tag">${kw}</span>`).join('')}
            </div>
            <div class="cluster-documents">
                ${cluster.documents.map(doc => createDocumentCard(doc)).join('')}
            </div>
        `;
        
        clustersView.appendChild(clusterDiv);
    });
}

// Display List
function displayList(results) {
    results.forEach(doc => {
        const docDiv = document.createElement('div');
        docDiv.innerHTML = createDocumentCard(doc);
        listView.appendChild(docDiv);
    });
}

// Display Multi-Modal Info
function displayMultimodalInfo(result) {
    const infoPanel = document.getElementById('multimodalInfo');
    const breakdown = document.getElementById('searchBreakdown');
    
    if (result.search_breakdown) {
        breakdown.innerHTML = `
            <div class="breakdown-item">
                <span class="breakdown-label">Text Match Score:</span>
                <span class="breakdown-value">${(result.search_breakdown.text_score || 0).toFixed(3)}</span>
            </div>
            <div class="breakdown-item">
                <span class="breakdown-label">Metadata Match Score:</span>
                <span class="breakdown-value">${(result.search_breakdown.metadata_score || 0).toFixed(3)}</span>
            </div>
            <div class="breakdown-item">
                <span class="breakdown-label">Combined Score (70% text + 30% metadata):</span>
                <span class="breakdown-value highlight">${(result.search_breakdown.combined_score || 0).toFixed(3)}</span>
            </div>
        `;
        infoPanel.classList.remove('hidden');
    }
}

// Create Document Card
function createDocumentCard(doc) {
    const metadata = doc.metadata || {};
    const title = metadata.title || doc.file_name || 'Untitled';
    const authors = metadata.authors ? metadata.authors.join(', ') : 'Unknown';
    const year = metadata.year || 'N/A';
    const abstract = metadata.abstract || doc.full_text || 'No abstract available';
    const score = doc.score !== undefined ? (typeof doc.score === 'number' ? doc.score.toFixed(3) : doc.score) : 'N/A';
    
    // Show which fields matched (metadata usage)
    const metadataMatched = [];
    if (metadata.authors && metadata.authors.length > 0) metadataMatched.push('Authors');
    if (metadata.keywords && metadata.keywords.length > 0) metadataMatched.push('Keywords');
    if (metadata.venue) metadataMatched.push('Venue');
    
    return `
        <div class="document-card">
            <div class="document-score">
                Relevance: ${score}
                ${metadataMatched.length > 0 ? `<span class="metadata-badge" title="This paper was found using metadata">📋 Metadata Used</span>` : ''}
            </div>
            <div class="document-title">${title}</div>
            <div class="document-meta">
                <span>👤 ${authors}</span>
                <span>📅 ${year}</span>
                ${metadata.venue ? `<span>📍 ${metadata.venue}</span>` : ''}
                ${metadata.keywords && metadata.keywords.length > 0 ? `<span>🏷️ Keywords: ${metadata.keywords.slice(0, 3).join(', ')}</span>` : ''}
            </div>
            <div class="document-abstract">${abstract.substring(0, 200)}...</div>
            ${metadataMatched.length > 0 ? `<div class="metadata-indicator">✨ Found using: ${metadataMatched.join(', ')}</div>` : ''}
        </div>
    `;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Check API health
    apiCall('/health', 'GET').then(result => {
        if (result.status === 'healthy') {
            showStatus('Connected to API server', 'success');
        }
    }).catch(() => {
        showStatus('Warning: Could not connect to API server. Make sure backend is running on port 5000.', 'error');
    });
});

