package main

import (
	"encoding/json"
	"fmt"
	"html/template"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
)

// ImageInfo represents information about a saved image
type ImageInfo struct {
	Filename  string    `json:"filename"`
	Path      string    `json:"path"`
	Timestamp time.Time `json:"timestamp"`
	Size      int64     `json:"size"`
}

// WebServer handles web interface for viewing images
type WebServer struct {
	outputDir string
	port      int
}

// NewWebServer creates a new web server
func NewWebServer(outputDir string, port int) *WebServer {
	return &WebServer{
		outputDir: outputDir,
		port:      port,
	}
}

// Start starts the web server
func (ws *WebServer) Start() error {
	router := mux.NewRouter()

	// Routes
	router.HandleFunc("/", ws.handleIndex).Methods("GET")
	router.HandleFunc("/api/images", ws.handleAPIImages).Methods("GET")
	router.HandleFunc("/image/{filename}", ws.handleImage).Methods("GET")
	router.PathPrefix("/static/").Handler(http.StripPrefix("/static/", http.FileServer(http.Dir("./static/"))))

	log.Printf("Starting web server on port %d", ws.port)
	log.Printf("Access web interface at: http://localhost:%d", ws.port)

	return http.ListenAndServe(fmt.Sprintf(":%d", ws.port), router)
}

// handleIndex serves the main page
func (ws *WebServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	page := r.URL.Query().Get("page")
	if page == "" {
		page = "1"
	}

	tmpl := `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cam-Stream Image Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .image-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .image-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .image-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            cursor: pointer;
        }
        .image-info {
            padding: 15px;
            background: #f9f9f9;
        }
        .image-info h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #555;
        }
        .image-info p {
            margin: 5px 0;
            font-size: 12px;
            color: #777;
        }
        .pagination {
            text-align: center;
            margin-top: 30px;
        }
        .pagination a {
            display: inline-block;
            padding: 8px 16px;
            margin: 0 4px;
            background: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            transition: background 0.2s;
        }
        .pagination a:hover {
            background: #0056b3;
        }
        .pagination .current {
            background: #6c757d;
        }
        .loading {
            text-align: center;
            padding: 50px;
            font-size: 18px;
            color: #666;
        }
        .no-images {
            text-align: center;
            padding: 50px;
            color: #666;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }
        .modal-content {
            display: block;
            margin: auto;
            max-width: 90%;
            max-height: 90%;
            margin-top: 50px;
        }
        .close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        .close:hover {
            color: #bbb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Cam-Stream Image Viewer</h1>
        
        <div id="loading" class="loading">Loading images...</div>
        <div id="content" style="display: none;">
            <div id="image-grid" class="image-grid"></div>
            <div id="pagination" class="pagination"></div>
        </div>
        <div id="no-images" class="no-images" style="display: none;">
            No detection images found. Images are only saved when objects are detected.
        </div>
    </div>

    <!-- Modal for full-size image -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        let currentPage = parseInt(new URLSearchParams(window.location.search).get('page') || '1');
        const imagesPerPage = 12;

        async function loadImages() {
            try {
                const response = await fetch('/api/images?page=' + currentPage + '&limit=' + imagesPerPage);
                const data = await response.json();
                
                document.getElementById('loading').style.display = 'none';
                
                if (data.images.length === 0) {
                    document.getElementById('no-images').style.display = 'block';
                    return;
                }
                
                displayImages(data.images);
                displayPagination(data.total, currentPage);
                document.getElementById('content').style.display = 'block';
            } catch (error) {
                console.error('Error loading images:', error);
                document.getElementById('loading').innerHTML = 'Error loading images: ' + error.message;
            }
        }

        function displayImages(images) {
            const grid = document.getElementById('image-grid');
            grid.innerHTML = '';
            
            images.forEach(image => {
                const card = document.createElement('div');
                card.className = 'image-card';
                
                const date = new Date(image.timestamp).toLocaleString();
                const sizeMB = (image.size / 1024 / 1024).toFixed(2);
                
                card.innerHTML = ` + "`" + `
                    <img src="/image/${image.filename}" alt="${image.filename}" onclick="openModal('/image/${image.filename}')">
                    <div class="image-info">
                        <h3>${image.filename}</h3>
                        <p><strong>Date:</strong> ${date}</p>
                        <p><strong>Size:</strong> ${sizeMB} MB</p>
                    </div>
                ` + "`" + `;
                
                grid.appendChild(card);
            });
        }

        function displayPagination(total, current) {
            const totalPages = Math.ceil(total / imagesPerPage);
            const pagination = document.getElementById('pagination');
            pagination.innerHTML = '';
            
            if (totalPages <= 1) return;
            
            // Previous button
            if (current > 1) {
                const prev = document.createElement('a');
                prev.href = '?page=' + (current - 1);
                prev.textContent = '← Previous';
                pagination.appendChild(prev);
            }
            
            // Page numbers
            for (let i = Math.max(1, current - 2); i <= Math.min(totalPages, current + 2); i++) {
                const link = document.createElement('a');
                link.href = '?page=' + i;
                link.textContent = i;
                if (i === current) {
                    link.className = 'current';
                }
                pagination.appendChild(link);
            }
            
            // Next button
            if (current < totalPages) {
                const next = document.createElement('a');
                next.href = '?page=' + (current + 1);
                next.textContent = 'Next →';
                pagination.appendChild(next);
            }
        }

        function openModal(imageSrc) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = imageSrc;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        // Close modal when clicking outside the image
        window.onclick = function(event) {
            const modal = document.getElementById('imageModal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }

        // Load images on page load
        loadImages();
    </script>
</body>
</html>`

	t, err := template.New("index").Parse(tmpl)
	if err != nil {
		http.Error(w, "Template error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html")
	t.Execute(w, nil)
}

// handleAPIImages returns JSON list of images
func (ws *WebServer) handleAPIImages(w http.ResponseWriter, r *http.Request) {
	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if page < 1 {
		page = 1
	}
	
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit < 1 || limit > 100 {
		limit = 12
	}

	images, err := ws.getImages()
	if err != nil {
		http.Error(w, "Error reading images: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Calculate pagination
	total := len(images)
	start := (page - 1) * limit
	end := start + limit
	
	if start >= total {
		start = 0
		end = 0
	}
	if end > total {
		end = total
	}

	var pageImages []ImageInfo
	if start < end {
		pageImages = images[start:end]
	}

	response := map[string]interface{}{
		"images": pageImages,
		"total":  total,
		"page":   page,
		"limit":  limit,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleImage serves individual images
func (ws *WebServer) handleImage(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	filename := vars["filename"]

	// Security check - prevent directory traversal
	if strings.Contains(filename, "..") || strings.Contains(filename, "/") || strings.Contains(filename, "\\") {
		http.Error(w, "Invalid filename", http.StatusBadRequest)
		return
	}

	imagePath := filepath.Join(ws.outputDir, filename)
	
	// Check if file exists
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		http.Error(w, "Image not found", http.StatusNotFound)
		return
	}

	http.ServeFile(w, r, imagePath)
}

// getImages returns list of all images sorted by modification time (newest first)
func (ws *WebServer) getImages() ([]ImageInfo, error) {
	files, err := ioutil.ReadDir(ws.outputDir)
	if err != nil {
		return nil, err
	}

	var images []ImageInfo
	for _, file := range files {
		if file.IsDir() {
			continue
		}

		// Only include image files
		filename := file.Name()
		ext := strings.ToLower(filepath.Ext(filename))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			continue
		}

		images = append(images, ImageInfo{
			Filename:  filename,
			Path:      filepath.Join(ws.outputDir, filename),
			Timestamp: file.ModTime(),
			Size:      file.Size(),
		})
	}

	// Sort by timestamp (newest first)
	sort.Slice(images, func(i, j int) bool {
		return images[i].Timestamp.After(images[j].Timestamp)
	})

	return images, nil
}
