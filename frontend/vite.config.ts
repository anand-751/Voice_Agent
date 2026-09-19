import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
	plugins: [react(), tailwindcss()],
	server: {
		host: '0.0.0.0',
		port: 5173,
		allowedHosts: true,
		proxy: {
			'/conversation': {
				target: 'http://localhost:8000',
				ws: true,
			},
			'/audio': {
				target: 'http://localhost:8000',
			},
			'/store-profile': {
				target: 'http://localhost:8000',
			},
			'/api': {
				target: 'http://localhost:8000',
			},
			'/health': {
				target: 'http://localhost:8000',
			},
		},
	},
})
