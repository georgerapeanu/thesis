/** @type {import('tailwindcss').Config} */
const colors = require('tailwindcss/colors');
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
    "./src/**/**/.{html,ts}",
  ],
  theme: {
    extend: {
      backgroundImage: {
        'attacked-piece': `radial-gradient(transparent 75%, ${colors['green'][900]} 75%)`,
        'in-check': `radial-gradient(${colors['red'][600]} 45%, transparent 80%)`
      }
    },
    colors: {
      ...colors,
      'chessBlackCell': '#b58863',
      'chessWhiteCell': '#f0d9b5'
    }
  },
  plugins: [],
}
