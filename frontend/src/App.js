import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box } from '@mui/material';

// Layout components
import Layout from './components/Layout';

// Pages
import Dashboard from './pages/Dashboard';
import Bots from './pages/Bots';
import CreateBot from './pages/CreateBot';
import BotDetails from './pages/BotDetails';
import Backtest from './pages/Backtest';
import Models from './pages/Models';
import Settings from './pages/Settings';
import NotFound from './pages/NotFound';

function App() {
  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="bots" element={<Bots />} />
          <Route path="bots/create" element={<CreateBot />} />
          <Route path="bots/:id" element={<BotDetails />} />
          <Route path="backtest" element={<Backtest />} />
          <Route path="models" element={<Models />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Box>
  );
}

export default App;
