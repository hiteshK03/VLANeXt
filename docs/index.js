document.addEventListener('DOMContentLoaded', () => {
    const roadmap = document.getElementById('roadmap-video');
    if (roadmap) {
        roadmap.load();
        roadmap.play().catch(error => {
            console.log("Autoplay was prevented:", error);
        });
    }

    const deferredVideos = Array.from(document.querySelectorAll('video'))
        .filter(v => v.id !== 'roadmap-video');

    const loadVideo = (video) => {
        const sources = video.querySelectorAll('source[data-src]');
        if (!sources.length) return;
        sources.forEach(source => {
            source.src = source.dataset.src;
            source.removeAttribute('data-src');
        });
        video.load();
        video.play().catch(error => {
            console.log("Autoplay was prevented:", error);
        });
    };

    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    loadVideo(entry.target);
                    obs.unobserve(entry.target);
                }
            });
        }, { rootMargin: '200px' });

        deferredVideos.forEach(video => observer.observe(video));
    } else {
        deferredVideos.forEach(video => {
            video.addEventListener('click', () => loadVideo(video), { once: true });
        });
    }
});
