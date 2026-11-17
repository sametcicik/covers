import react from '@vitejs/plugin-react';
import {defineConfig} from 'vite';

export default defineConfig({
	root: 'src/ui',
	plugins: [react()],
	build: {outDir: '../../dist', emptyOutDir: true, assetsDir: 'ui-assets', chunkSizeWarningLimit: 2000},
	server: {
		proxy: {
			'/api': 'http://127.0.0.1:8765',
			'/work': 'http://127.0.0.1:8765',
			'/assets': 'http://127.0.0.1:8765',
			'/streamers': 'http://127.0.0.1:8765',
		},
	},
});
