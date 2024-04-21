/** @type {import('tailwindcss').Config} */
const colors = require('tailwindcss/colors');
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
    "./src/**/**/.{html,ts}",
  ],
  theme: {
    extend: {},
    colors: {
      ...colors,
      'chessBlackCell': '#b58863',
      'chessWhiteCell': '#f0d9b5'
    }
  },
  plugins: [],
}
