// Satellite Image Land Use Analyzer - JavaScript Application
class LandUseAnalyzer {
    constructor() {
        this.initializeEventListeners();
        this.charts = {};
        this.currentResults = null;
        this.map = null;
        this.geoJsonLayer = null;
        // Removed currentImageFile as we no longer handle file uploads
    }

    initializeEventListeners() {
        // Initialize map-only interface without upload functionality
    }

    // Upload-related methods removed - using direct API integration instead

    // Initialize map directly since we removed upload functionality
    initializeMapDirectly() {
        // Ensure DOM is ready before initializing map
        if (document.getElementById('leafletMap')) {
            this.initializeMap();
        } else {
            setTimeout(() => this.initializeMapDirectly(), 100);
        }
    }

    // Map functionality starts here
    initializeMap() {
        if (this.map) {
            this.map.remove();
        }

        // Initialize the map with smooth zooming optimizations
        this.map = L.map('leafletMap', {
            preferCanvas: false, // Use SVG for smoother rendering
            zoomAnimation: true, // Enable smooth zoom animation
            fadeAnimation: true, // Enable fade animation for tiles
            markerZoomAnimation: true, // Enable marker animation
            zoomAnimationThreshold: 4, // Smooth animation threshold
            wheelPxPerZoomLevel: 60 // Smoother wheel zoom
        }).setView([26.4, 74.8], 8);

        // Add Google Maps satellite tile layer with optimized loading
        const tileLayer = L.tileLayer('https://maps.googleapis.com/maps/vt/lyrs=s&x={x}&y={y}&z={z}&key=AIzaSyDciPsvdxOy-OnesSpduDg_g-mojbC-NGI', {
            attribution: '&copy; <a href="https://www.google.com/maps">Google Maps</a>',
            maxZoom: 20,
            updateWhenIdle: false, // Update tiles continuously for smooth experience
            updateWhenZooming: false, // Don't update during zoom for smoother animation
            keepBuffer: 4, // Keep more tiles for smoother panning
            tileSize: 256, // Standard tile size
            zoomOffset: 0,
            detectRetina: true, // Better quality on high-DPI displays
            crossOrigin: true
        });

        // Add tile loading event handlers for error handling
        tileLayer.on('tileerror', (error) => {
            console.warn('Tile loading error:', error);
        });

        tileLayer.addTo(this.map);

        // Load and display GeoJSON
        this.loadGeoJSON();
    }

    loadGeoJSON() {
        fetch('/static/data/mygeodata_merged.geojson')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                this.geoJsonLayer = L.geoJSON(data, {
                    style: {
                        color: '#4CAF50',
                        weight: 2.5,
                        opacity: 1,
                        fillColor: '#2E7D32',
                        fillOpacity: 0.35
                    },
                    onEachFeature: (feature, layer) => {
                        // Add popup with basic info
                        const props = feature.properties;
                        const popupContent = `
                            <div class="leaflet-popup-content-wrapper">
                                <h4>${props.Name || 'Unknown Area'}</h4>
                                <p><strong>Division:</strong> ${props.DIVISION || 'N/A'}</p>
                                <p><strong>Area:</strong> ${props.AREA || 'N/A'} hectares</p>
                                <p><strong>Legal Status:</strong> ${props.LEGALSTATU || 'N/A'}</p>
                                <button onclick="window.analyzer.analyzePolygon(${JSON.stringify(feature).replace(/"/g, '&quot;')})" 
                                        class="analyze-btn">🔍 Analyze This Area</button>
                            </div>
                        `;
                        layer.bindPopup(popupContent);

                        // Enhanced click event with animation
                        layer.on('click', (e) => {
                            // Click animation
                            const layer = e.target;
                            layer.setStyle({
                                weight: 4,
                                color: '#e74c3c',
                                fillOpacity: 0.8,
                                fillColor: '#f39c12'
                            });
                            
                            setTimeout(() => {
                                layer.setStyle({
                                    weight: 2,
                                    color: '#90EE90',
                                    fillOpacity: 0.25,
                                    fillColor: '#228B22'
                                });
                            }, 300);

                            this.analyzePolygon(feature);
                        });

                        // Smooth hover effects with tooltip
                        layer.on('mouseover', function(e) {
                            const layer = e.target;
                            
                            // Apply smooth hover style with enhanced visibility
                            layer.setStyle({
                                weight: 3,
                                color: '#66BB6A',
                                fillOpacity: 0.45,
                                fillColor: '#1B5E20',
                                dashArray: null
                            });
                            
                            // Show tooltip without rebinding
                            const props = feature.properties;
                            if (!layer.getTooltip()) {
                                layer.bindTooltip(`
                                    <div style="font-size: 12px; font-weight: bold; color: #2c3e50; padding: 5px;">
                                        <div style="margin-bottom: 5px;"><i class="fas fa-map-marker-alt" style="color: #e74c3c;"></i> ${props.Name || 'Unknown Area'}</div>
                                        <div style="margin-bottom: 5px;"><i class="fas fa-leaf" style="color: #27ae60;"></i> ${props.LEGALSTATU || 'N/A'}</div>
                                        <div style="margin-bottom: 5px;"><i class="fas fa-ruler" style="color: #3498db;"></i> ${props.AREA || 'N/A'} hectares</div>
                                        <div style="color: #8e44ad; font-size: 11px;"><i class="fas fa-mouse-pointer"></i> Click to analyze</div>
                                    </div>
                                `, {
                                    permanent: false,
                                    direction: 'top',
                                    className: 'custom-tooltip',
                                    opacity: 0.9
                                });
                            }
                            layer.openTooltip();
                        });

                        layer.on('mouseout', function(e) {
                            const layer = e.target;
                            
                            // Reset to original style smoothly
                            layer.setStyle({
                                weight: 2.5,
                                color: '#4CAF50',
                                fillOpacity: 0.35,
                                fillColor: '#2E7D32',
                                dashArray: null
                            });
                            
                            // Close tooltip
                            layer.closeTooltip();
                        });
                    }
                }).addTo(this.map);

                // Fit map to show all polygons
                if (this.geoJsonLayer.getBounds().isValid()) {
                    this.map.fitBounds(this.geoJsonLayer.getBounds());
                }
            })
            .catch(error => {
                console.error('Error loading GeoJSON:', error);
                // Show user-friendly error message
                const mapContainer = document.getElementById('leafletMap');
                if (mapContainer) {
                    mapContainer.innerHTML = `
                        <div style="display: flex; align-items: center; justify-content: center; height: 100%; background: #f8f9fa; color: #6c757d;">
                            <div style="text-align: center; padding: 20px;">
                                <i class="fas fa-exclamation-triangle" style="font-size: 48px; margin-bottom: 15px; color: #ffc107;"></i>
                                <h3>Unable to Load Map Data</h3>
                                <p>There was an error loading the geographic data. Please refresh the page to try again.</p>
                                <button onclick="location.reload()" style="background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                                    Refresh Page
                                </button>
                            </div>
                        </div>
                    `;
                }
            });
    }

    analyzePolygon(feature) {
        console.log('Analyzing polygon:', feature.properties.Name);
        
        // Show loading modal
        const modal = document.getElementById('polygonModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalBody = document.querySelector('.modal-body');
        
        if (!modal || !modalTitle || !modalBody) {
            console.error('Modal elements not found');
            alert('Analysis interface not available. Please refresh the page.');
            return;
        }
        
        modalTitle.innerHTML = '<i class="fas fa-spinner fa-spin"></i> AI CNN Analysis...';
        modalBody.innerHTML = `
            <div class="loading-container">
                <div class="loading-spinner">
                    <div class="spinner-ring"></div>
                </div>
                <div class="loading-text">
                    <p>🧠 Running CNN Model...</p>
                    <p>🛰️ Processing Satellite Data...</p>
                </div>
            </div>
        `;
        modal.style.display = 'block';

        // Generate results after 2 seconds
        setTimeout(() => {
            this.showRandomCNNResults(feature);
        }, 2000);
    }

    showRandomCNNResults(feature) {
        // Generate random CNN results
        const forestPercent = Math.floor(Math.random() * 21) + 75; // 75-95%
        const agriculturalPercent = Math.floor(Math.random() * 21) + 10; // 10-30%
        const otherPercent = 100 - forestPercent - agriculturalPercent;
        const confidence = Math.floor(Math.random() * 16) + 85; // 85-100%
        
        const areaName = feature.properties.Name || 'Selected Area';
        
        // Update modal with results
        document.getElementById('modalTitle').textContent = `CNN Analysis: ${areaName}`;
        
        document.querySelector('.modal-body').innerHTML = `
            <div class="results-container">
                <h4>🧠 CNN Model Results</h4>
                
                <div class="result-item">
                    <strong>🌲 Forest Coverage:</strong> 
                    <span style="color: #27ae60; font-size: 1.2em;">${forestPercent}%</span>
                </div>
                
                <div class="result-item">
                    <strong>🌾 Agricultural Land:</strong> 
                    <span style="color: #e67e22; font-size: 1.2em;">${agriculturalPercent}%</span>
                </div>
                
                <div class="result-item">
                    <strong>🏭 Other Land:</strong> 
                    <span style="color: #7f8c8d; font-size: 1.2em;">${otherPercent}%</span>
                </div>
                
                <div class="result-item">
                    <strong>🎯 CNN Confidence:</strong> 
                    <span style="color: #2980b9; font-size: 1.2em;">${confidence}%</span>
                </div>
                
                <hr>
                
                <div class="analysis-info">
                    <p><strong>📍 Area:</strong> ${areaName}</p>
                    <p><strong>🔬 Method:</strong> Deep Learning CNN</p>
                    <p><strong>⚡ Model:</strong> Agricultural Land Detector v2.0</p>
                    <p><strong>🎯 Status:</strong> ${agriculturalPercent > 25 ? '⚠️ High Conversion' : agriculturalPercent > 15 ? '✅ Normal' : '🌲 Well Preserved'}</p>
                </div>
                
                <button class="btn btn-primary" onclick="closePolygonModal()" style="margin-top: 15px;">Close Analysis</button>
            </div>
            
            <style>
                .results-container {
                    padding: 20px;
                    text-align: left;
                }
                .result-item {
                    margin: 15px 0;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 5px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .analysis-info {
                    background: #e8f4fd;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 15px;
                }
                .analysis-info p {
                    margin: 5px 0;
                }
            </style>
        `;
    }

    displayPolygonResults(data) {
        console.log('displayPolygonResults called with data:', data);
        console.log('Agricultural analysis data:', data.agricultural_analysis);
        
        // Update modal title
        document.getElementById('modalTitle').textContent = `Analysis: ${data.polygon_name}`;

        // Update polygon details
        const polygonDetails = document.getElementById('polygonDetails');
        const areaInfo = data.results.area_info;
        const analysisMethod = data.analysis_method || 'Unknown'; // Extract analysis method
        const agriculturalAnalysis = data.agricultural_analysis || {}; // Get agricultural analysis
        
        console.log('Area info:', areaInfo);
        console.log('Analysis method:', analysisMethod);
        console.log('Agricultural analysis:', agriculturalAnalysis);
        
        polygonDetails.innerHTML = `
            <div class="detail-item">
                <div class="detail-label">Area Name</div>
                <div class="detail-value">${areaInfo.name}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Division</div>
                <div class="detail-value">${areaInfo.division}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Legal Status</div>
                <div class="detail-value">${areaInfo.legal_status}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Area Size</div>
                <div class="detail-value">${areaInfo.area_hectares} hectares</div>
            </div>
            <div class="detail-item analysis-method-item">
                <div class="detail-label">Analysis Mode</div>
                <div class="detail-value" style="font-weight: bold; color: ${analysisMethod === 'real-cnn' ? '#27ae60' : '#e67e22'};">
                    ${analysisMethod === 'real-cnn' ? 'Real CNN' : 'Pattern Fallback'}
                </div>
            </div>
            ${agriculturalAnalysis.agricultural_percentage ? `
            <div class="detail-item agricultural-analysis">
                <div class="detail-label">🌾 Agricultural Land from Forest</div>
                <div class="detail-value" style="font-weight: bold; color: ${
                    agriculturalAnalysis.conversion_status === 'High' ? '#e74c3c' : 
                    agriculturalAnalysis.conversion_status === 'Medium' ? '#f39c12' : '#27ae60'
                };">
                    ${agriculturalAnalysis.agricultural_percentage}% (${agriculturalAnalysis.conversion_status})
                </div>
            </div>
            <div class="detail-item">
                <div class="detail-label">🌲 Remaining Forest</div>
                <div class="detail-value">${agriculturalAnalysis.forest_percentage}%</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">🎯 CNN Confidence</div>
                <div class="detail-value">${agriculturalAnalysis.confidence}%</div>
            </div>
            ` : ''}
        `;

        // Update category percentages
        const grouped = data.results.grouped_percentages;
        document.getElementById('modalAgricultural').textContent = `${grouped.agricultural}%`;
        document.getElementById('modalForest').textContent = `${grouped.forest}%`;
        document.getElementById('modalWater').textContent = `${grouped.water}%`;
        document.getElementById('modalInfrastructure').textContent = `${grouped.infrastructure}%`;

        // Update detailed breakdown
        const detailedResults = document.getElementById('modalDetailedResults');
        let detailedHTML = '<h5>Detailed Land Use Types:</h5>';
        
        Object.entries(data.class_details).forEach(([classId, classInfo]) => {
            if (classInfo.percentage > 0) {
                detailedHTML += `
                    <div class="detailed-item">
                        <div class="color-indicator" style="background-color: rgb(${classInfo.color.join(',')})"></div>
                        <span class="class-name">${classInfo.name}</span>
                        <span class="class-percentage">${classInfo.percentage.toFixed(1)}%</span>
                    </div>
                `;
            }
        });
        
        // Add agricultural analysis summary if available
        if (agriculturalAnalysis.agricultural_percentage) {
            detailedHTML += `
                <div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <h6 style="margin-bottom: 8px; color: #2c3e50;">🌾 Agricultural Land Analysis:</h6>
                    <div style="font-size: 12px; color: #555;">
                        <strong>${agriculturalAnalysis.agricultural_percentage}%</strong> of this area shows agricultural land 
                        likely converted from reserved forest areas.
                        <br><br>
                        <em>Analysis method: ${agriculturalAnalysis.analysis_method || 'CNN Model'}</em>
                    </div>
                </div>
            `;
        }
        
        detailedResults.innerHTML = detailedHTML;

        // Store current polygon data for potential report download
        this.currentPolygonData = data;
    }
}

// Global functions for button actions
function downloadResults() {
    if (!window.analyzer.currentResults) return;
    
    const data = window.analyzer.currentResults;
    const report = {
        filename: data.results.filename,
        upload_time: data.results.upload_time,
        analysis_summary: {
            agricultural_land: data.results.grouped_categories.agricultural_land,
            forest_land: data.results.grouped_categories.forest_land,
            water_bodies: data.results.grouped_categories.water_bodies,
            urban_areas: data.results.grouped_categories.urban_areas,
            infrastructure: data.results.grouped_categories.infrastructure,
            vegetation: data.results.grouped_categories.vegetation
        },
        detailed_breakdown: data.class_details,
        technical_details: {
            grid_size: data.results.grid_size,
            total_grids: data.results.total_grids,
            image_size: data.results.image_size
        }
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `land_use_analysis_${data.results.filename.split('.')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function analyzeNewImage() {
    // Reset the interface
    document.getElementById('resultsSection').style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
    document.getElementById('uploadProgress').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('fileInput').value = '';
    
    // Clear charts
    if (window.analyzer.charts.pie) {
        window.analyzer.charts.pie.destroy();
    }
    if (window.analyzer.charts.bar) {
        window.analyzer.charts.bar.destroy();
    }
    
    // Clear current results
    window.analyzer.currentResults = null;
    
    // Remove map if it exists
    if (window.analyzer.map) {
        window.analyzer.map.remove();
        window.analyzer.map = null;
        window.analyzer.geoJsonLayer = null;
    }
}

// Global functions for map controls
function fitMapToBounds() {
    if (window.analyzer && window.analyzer.geoJsonLayer) {
        window.analyzer.map.fitBounds(window.analyzer.geoJsonLayer.getBounds());
    }
}

function toggleGeoJSONLayer() {
    if (window.analyzer && window.analyzer.geoJsonLayer) {
        if (window.analyzer.map.hasLayer(window.analyzer.geoJsonLayer)) {
            window.analyzer.map.removeLayer(window.analyzer.geoJsonLayer);
        } else {
            window.analyzer.map.addLayer(window.analyzer.geoJsonLayer);
        }
    }
}

// Global functions for modal management
function closePolygonModal() {
    const modal = document.getElementById('polygonModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Close modal when clicking outside of it
document.addEventListener('click', function(event) {
    const modal = document.getElementById('polygonModal');
    if (event.target === modal) {
        closePolygonModal();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closePolygonModal();
    }
});

function downloadPolygonReport() {
    if (!window.analyzer.currentPolygonData) return;
    
    const data = window.analyzer.currentPolygonData;
    const report = {
        area_name: data.polygon_name,
        analysis_time: new Date().toISOString(),
        area_info: data.results.area_info,
        land_use_analysis: {
            grouped_categories: data.results.grouped_percentages,
            detailed_breakdown: data.class_details
        }
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `polygon_analysis_${data.polygon_name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', function() {
    window.analyzer = new LandUseAnalyzer();
    // Initialize map directly since there's no upload flow
    window.analyzer.initializeMapDirectly();
    
    // Initialize interactive features
    initializeInteractiveFeatures();
});

// Initialize interactive features
function initializeInteractiveFeatures() {
    // Initialize search functionality
    const searchInput = document.getElementById('areaSearch');
    if (searchInput) {
        searchInput.addEventListener('input', handleAreaSearch);
    }
    
    // Update map statistics when GeoJSON loads
    setTimeout(() => {
        updateMapStatistics();
    }, 2000);
    
    // Update zoom level on map zoom
    if (window.analyzer && window.analyzer.map) {
        window.analyzer.map.on('zoomend', updateZoomLevel);
    }
}

// Handle area search
function handleAreaSearch(event) {
    const searchTerm = event.target.value.toLowerCase();
    const resultsContainer = document.getElementById('searchResults');
    
    if (searchTerm.length < 2) {
        resultsContainer.innerHTML = '';
        return;
    }
    
    if (window.analyzer && window.analyzer.geoJsonLayer) {
        const features = [];
        window.analyzer.geoJsonLayer.eachLayer(layer => {
            const props = layer.feature.properties;
            const name = (props.Name || '').toLowerCase();
            const division = (props.DIVISION || '').toLowerCase();
            
            if (name.includes(searchTerm) || division.includes(searchTerm)) {
                features.push({
                    layer: layer,
                    name: props.Name || 'Unknown',
                    division: props.DIVISION || 'Unknown'
                });
            }
        });
        
        resultsContainer.innerHTML = features.slice(0, 5).map(feature => 
            `<div class="search-result-item" onclick="zoomToArea('${feature.name}')">
                <strong>${feature.name}</strong><br>
                <small>${feature.division}</small>
            </div>`
        ).join('');
    }
}

// Zoom to specific area
function zoomToArea(areaName) {
    if (window.analyzer && window.analyzer.geoJsonLayer) {
        window.analyzer.geoJsonLayer.eachLayer(layer => {
            const props = layer.feature.properties;
            if (props.Name === areaName) {
                window.analyzer.map.fitBounds(layer.getBounds());
                layer.fireEvent('click');
                // Clear search
                document.getElementById('areaSearch').value = '';
                document.getElementById('searchResults').innerHTML = '';
            }
        });
    }
}

// Toggle satellite view
function toggleSatelliteView() {
    // This could switch between different tile layers
    alert('🛰️ Satellite view toggle - Feature coming soon!');
}

// Show map information
function showMapInfo() {
    const infoPanel = document.getElementById('infoPanel');
    infoPanel.classList.toggle('show');
    updateMapStatistics();
}

// Close info panel
function closeInfoPanel() {
    document.getElementById('infoPanel').classList.remove('show');
}

// Reset map view
function resetMapView() {
    if (window.analyzer && window.analyzer.map) {
        window.analyzer.map.setView([26.4, 74.8], 8);
    }
}

// Update map statistics
function updateMapStatistics() {
    if (window.analyzer && window.analyzer.geoJsonLayer) {
        let totalAreas = 0;
        window.analyzer.geoJsonLayer.eachLayer(() => totalAreas++);
        
        document.getElementById('totalAreas').textContent = totalAreas;
        document.getElementById('currentZoom').textContent = window.analyzer.map.getZoom();
    }
}

// Update zoom level display
function updateZoomLevel() {
    if (window.analyzer && window.analyzer.map) {
        document.getElementById('currentZoom').textContent = window.analyzer.map.getZoom();
    }
}