(function ($) {
    "use strict";

    // Loader
    var loader = function () {
        setTimeout(function () {
            if ($('#loader').length > 0) {
                $('#loader').removeClass('show');
            }
        }, 1);
    };
    loader();
    
    // Initiate the wowjs
    new WOW().init();
    
    // Smooth scrolling on the navbar links
    $(".navbar-nav a").on('click', function (event) {
        // Vérifiez si this.hash est défini et non vide
        if (this.hash !== "" && $(this.hash).length) {
            event.preventDefault(); // Empêche le comportement par défaut du lien
            
            // Stockez le hash
            var hash = this.hash;
            
            // Animez le défilement fluide
            $('html, body').animate({
                scrollTop: $(hash).offset().top - 45
            }, 800, 'easeInOutExpo', function () {
                // Ajoutez le hash à l'URL une fois le défilement terminé (comportement par défaut du clic)
                window.location.hash = hash;
            });
            
            // Met en surbrillance le lien actif
            if ($(this).parents('.navbar-nav').length) {
                $('.navbar-nav .active').removeClass('active');
                $(this).addClass('active');
            }
        }
    });
    
    // Back to top button
    $(window).scroll(function () {
        if ($(this).scrollTop() > 200) {
            $('.back-to-top').fadeIn('slow');
        } else {
            $('.back-to-top').fadeOut('slow');
        }
    });
    $('.back-to-top').click(function () {
        $('html, body').animate({scrollTop: 0}, 1500, 'easeInOutExpo');
        return false;
    });
    
    // Sticky Navbar
    $(window).scroll(function () {
        if ($(this).scrollTop() > 0) {
            $('.navbar').addClass('nav-sticky');
        } else {
            $('.navbar').removeClass('nav-sticky');
        }
    });

    // Active class for current page
    $(document).ready(function() {
        // Obtenez le chemin d'URL actuel
        var pathname = window.location.pathname;

        // Mettez en surbrillance l'élément nav correct en fonction de la page actuelle
        if (pathname.includes("index.html")) {
            $('a[href^="index.html"]').addClass('active');
        } else if (pathname.includes("page1.html")) {
            $('a[href^="page1.html"]').addClass('active');
        } else if (pathname.includes("page2.html")) {
            $('a[href^="page2.html"]').addClass('active');
        }
        // Ajoutez plus de conditions si vous avez plus de pages
    });

    // Typed Initiate
    if ($('.hero .hero-text h2').length == 1) {
        var typed_strings = $('.hero .hero-text .typed-text').text();
        var typed = new Typed('.hero .hero-text h2', {
            strings: typed_strings.split(', '),
            typeSpeed: 100,
            backSpeed: 20,
            smartBackspace: false,
            loop: true
        });
    }
    
    // Skills
    $('.skills').waypoint(function () {
        $('.progress .progress-bar').each(function () {
            $(this).css("width", $(this).attr("aria-valuenow") + '%');
        });
    }, {offset: '80%'});

    // Testimonials carousel
    $(".testimonials-carousel").owlCarousel({
        center: true,
        autoplay: true,
        dots: true,
        loop: true,
        responsive: {
            0:{
                items:1
            }
        }
    });
    
    // Portfolio filter
    var portfolioIsotope = $('.portfolio-container').isotope({
        itemSelector: '.portfolio-item',
        layoutMode: 'fitRows'
    });

    $('#portfolio-filter li').on('click', function () {
        $("#portfolio-filter li").removeClass('filter-active');
        $(this).addClass('filter-active');
        portfolioIsotope.isotope({filter: $(this).data('filter')});
    });
    
})(jQuery);
