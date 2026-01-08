/**
 * Lightbox functionality for Sphinx documentation figures
 * Allows clicking on images to view them at full size in a modal overlay
 */

(function() {
    'use strict';

    // Create lightbox elements once DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        // Track scroll position
        let savedScrollPosition = 0;

        // Create the lightbox overlay structure
        const overlay = document.createElement('div');
        overlay.className = 'lightbox-overlay';
        overlay.innerHTML = `
            <button class="lightbox-close" aria-label="Close" type="button">&times;</button>
            <div class="lightbox-content">
                <img class="lightbox-image" src="" alt="Enlarged view">
            </div>
        `;
        document.body.appendChild(overlay);

        const lightboxImage = overlay.querySelector('.lightbox-image');
        const closeButton = overlay.querySelector('.lightbox-close');

        /**
         * Open the lightbox with the given image source
         * @param {string} src - The image source URL
         * @param {string} alt - The image alt text
         * @param {boolean} isSvg - Whether the image is an SVG
         */
        function openLightbox(src, alt, isSvg) {
            // Save current scroll position
            savedScrollPosition = window.pageYOffset || document.documentElement.scrollTop;
            
            // For SVGs, add a class to handle sizing differently
            if (isSvg) {
                lightboxImage.classList.add('lightbox-svg');
            } else {
                lightboxImage.classList.remove('lightbox-svg');
            }
            lightboxImage.src = src;
            lightboxImage.alt = alt || 'Enlarged view';
            overlay.classList.add('active');
            document.body.classList.add('lightbox-open');
        }

        /**
         * Close the lightbox
         */
        function closeLightbox() {
            overlay.classList.remove('active');
            document.body.classList.remove('lightbox-open');
            
            // Restore scroll position
            window.scrollTo(0, savedScrollPosition);
            
            // Clear the image source after transition
            setTimeout(function() {
                if (!overlay.classList.contains('active')) {
                    lightboxImage.src = '';
                    lightboxImage.classList.remove('lightbox-svg');
                }
            }, 300);
        }

        /**
         * Check if image source is an SVG
         * @param {string} src - The image source URL
         * @returns {boolean} True if SVG
         */
        function isSvgImage(src) {
            return src && src.toLowerCase().includes('.svg');
        }

        /**
         * Get the full-size image source
         * For Sphinx, images might be in _images/ directory in built docs
         * @param {HTMLImageElement} img - The image element
         * @returns {string} The full-size image URL
         */
        function getFullSizeSource(img) {
            // Use the original src - Sphinx typically serves full images
            return img.src;
        }

        /**
         * Check if an element should trigger lightbox
         * @param {HTMLElement} element - The element to check
         * @returns {boolean} True if element should trigger lightbox
         */
        function shouldTriggerLightbox(element) {
            // Skip if it's the lightbox image itself
            if (element.classList.contains('lightbox-image')) {
                return false;
            }
            // Skip logo images
            if (element.closest('.sidebar-brand') || element.closest('.navbar-brand')) {
                return false;
            }
            // Skip very small raster images (likely icons)
            // SVGs may not have naturalWidth/naturalHeight, so check file extension
            const src = element.src || '';
            if (!isSvgImage(src) && element.naturalWidth < 50 && element.naturalHeight < 50) {
                return false;
            }
            return true;
        }

        // Event delegation for image clicks
        document.addEventListener('click', function(e) {
            const target = e.target;
            
            // Check if clicked element is an image
            if (target.tagName === 'IMG' && shouldTriggerLightbox(target)) {
                e.preventDefault();
                e.stopPropagation();
                const src = getFullSizeSource(target);
                openLightbox(src, target.alt, isSvgImage(src));
                return false;
            }

            // Check if clicked inside a figure (but not on caption text)
            const figure = target.closest('figure, .figure');
            if (figure) {
                const img = figure.querySelector('img');
                if (img && shouldTriggerLightbox(img) && !target.closest('figcaption, .caption')) {
                    e.preventDefault();
                    e.stopPropagation();
                    const src = getFullSizeSource(img);
                    openLightbox(src, img.alt, isSvgImage(src));
                    return false;
                }
            }
        });

        // Close button click handler - prevent default and stop propagation
        closeButton.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            closeLightbox();
            return false;
        });

        // Close lightbox when clicking overlay background
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) {
                e.preventDefault();
                e.stopPropagation();
                closeLightbox();
                return false;
            }
        });

        // Also close when clicking the image itself (user expectation)
        lightboxImage.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            closeLightbox();
            return false;
        });

        // Close on Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && overlay.classList.contains('active')) {
                e.preventDefault();
                closeLightbox();
            }
        });
    });
})();
