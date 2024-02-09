const wrapper = document.querySelector(".wrapper");
const registerLink = document.querySelector(".register-link");
const loginLink = document.querySelector(".login-link");
const btnPopup = document.querySelector(".btnLogin-popup");
const iconClose = document.querySelector(".icon-close");

registerLink.onclick = () => {
    wrapper.classList.add('active');
};

loginLink.onclick = () => {
    wrapper.classList.remove('active');
};

btnPopup.onclick = () => {
    wrapper.classList.add('active-popup');
};

iconClose.onclick = () => {
    wrapper.classList.remove('active-popup');
    wrapper.classList.remove('active');
};

function validateEmail() {
    var emailInput = document.getElementById('emailInput');
    var emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!emailRegex.test(emailInput.value)) {
        alert('Please enter a valid email address.');
        return false;
    }
    return true;
}

// script.js

// Initialize ScrollReveal
ScrollReveal().reveal('#about-heading', {
    delay: 200,
    origin: 'bottom',
    distance: '50px',
    duration: 1000,
    easing: 'ease-in-out',
});

ScrollReveal().reveal('#about-content', {
    delay: 300,
    origin: 'bottom',
    distance: '50px',
    duration: 1000,
    easing: 'ease-in-out',
});

ScrollReveal().reveal('#services-heading', {
    delay: 200,
    origin: 'bottom',
    distance: '50px',
    duration: 1000,
    easing: 'ease-in-out',
});

ScrollReveal().reveal('#services-content', {
    delay: 300,
    origin: 'bottom',
    distance: '50px',
    duration: 1000,
    easing: 'ease-in-out',
});

ScrollReveal().reveal('#contact-heading', {
    delay: 200,
    origin: 'bottom',
    distance: '50px',
    duration: 1000,
    easing: 'ease-in-out',
});

ScrollReveal().reveal('#contact-content', {
    delay: 300,
    origin: 'bottom',
    distance: '50px',
    duration: 1000,
    easing: 'ease-in-out',
});

