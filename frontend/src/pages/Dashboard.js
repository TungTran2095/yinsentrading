import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { 
  Grid, 
  Paper, 
  Typography, 
  Box, 
  Card, 
  CardContent, 
  CardHeader,
  Button,
  Divider,
  CircularProgress
} from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { fetchBots } from '../store/botsSlice';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = () => {
  const dispatch = useDispatch();
  const { bots, loading } = useSelector((state) => state.bots);

  useEffect(() => {
    dispatch(fetchBots());
  }, [dispatch]);

  // Prepare data for charts
  const prepareEquityChartData = () => {
    if (!bots || bots.length === 0) return null;

    // Use the first bot with equity history for demo
    const bot = bots.find(b => b.account_info && b.account_info.equity);
    if (!bot) return null;

    return {
      labels: ['1d', '2d', '3d', '4d', '5d', '6d', '7d'],
      datasets: [
        {
          label: 'Account Equity',
          data: [
            bot.account_info.equity * 0.95,
            bot.account_info.equity * 0.97,
            bot.account_info.equity * 0.96,
            bot.account_info.equity * 0.98,
            bot.account_info.equity * 1.01,
            bot.account_info.equity * 1.02,
            bot.account_info.equity,
          ],
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1,
          fill: false,
        },
      ],
    };
  };

  const prepareProfitChartData = () => {
    return {
      labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
      datasets: [
        {
          label: 'Daily Profit',
          data: [12, -19, 3, 5, 2, 3, 7],
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
        },
      ],
    };
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Performance Chart',
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      },
    },
  };

  // Calculate summary stats
  const calculateStats = () => {
    if (!bots || bots.length === 0) return {
      totalBots: 0,
      activeBots: 0,
      totalEquity: 0,
      totalPnL: 0,
    };

    const activeBots = bots.filter(bot => bot.status === 'running').length;
    const totalEquity = bots.reduce((sum, bot) => 
      sum + (bot.account_info?.equity || 0), 0);
    const totalPnL = bots.reduce((sum, bot) => {
      const initialBalance = 10000; // Assuming default initial balance
      return sum + ((bot.account_info?.equity || 0) - initialBalance);
    }, 0);

    return {
      totalBots: bots.length,
      activeBots,
      totalEquity,
      totalPnL,
    };
  };

  const stats = calculateStats();
  const equityChartData = prepareEquityChartData();
  const profitChartData = prepareProfitChartData();

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom component="div">
        Dashboard
      </Typography>
      
      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 140,
              bgcolor: 'primary.dark',
              color: 'white',
            }}
          >
            <Typography component="h2" variant="h6" color="inherit" gutterBottom>
              Total Bots
            </Typography>
            <Typography component="p" variant="h4">
              {stats.totalBots}
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              {stats.activeBots} active
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 140,
              bgcolor: stats.totalPnL >= 0 ? 'success.dark' : 'error.dark',
              color: 'white',
            }}
          >
            <Typography component="h2" variant="h6" color="inherit" gutterBottom>
              Total P&L
            </Typography>
            <Typography component="p" variant="h4">
              ${stats.totalPnL.toFixed(2)}
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              {stats.totalPnL >= 0 ? '+' : ''}{(stats.totalPnL / (stats.totalEquity - stats.totalPnL) * 100).toFixed(2)}%
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 140,
              bgcolor: 'info.dark',
              color: 'white',
            }}
          >
            <Typography component="h2" variant="h6" color="inherit" gutterBottom>
              Total Equity
            </Typography>
            <Typography component="p" variant="h4">
              ${stats.totalEquity.toFixed(2)}
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              Across all bots
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 140,
              bgcolor: 'warning.dark',
              color: 'white',
            }}
          >
            <Typography component="h2" variant="h6" color="inherit" gutterBottom>
              Win Rate
            </Typography>
            <Typography component="p" variant="h4">
              58%
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              Last 30 days
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 400,
            }}
          >
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Equity Curve
            </Typography>
            {equityChartData ? (
              <Box sx={{ height: '100%', width: '100%' }}>
                <Line options={chartOptions} data={equityChartData} />
              </Box>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <Typography variant="body1" color="text.secondary">
                  No data available
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 400,
            }}
          >
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Daily Profit
            </Typography>
            <Box sx={{ height: '100%', width: '100%' }}>
              <Line options={chartOptions} data={profitChartData} />
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Active Bots */}
      <Box sx={{ mt: 4 }}>
        <Typography variant="h5" gutterBottom component="div">
          Active Bots
        </Typography>
        <Grid container spacing={3}>
          {bots.filter(bot => bot.status === 'running').length > 0 ? (
            bots.filter(bot => bot.status === 'running').map((bot) => (
              <Grid item xs={12} sm={6} md={4} key={bot.id}>
                <Card>
                  <CardHeader
                    title={bot.name}
                    subheader={`${bot.strategy} - ${bot.symbol}`}
                  />
                  <Divider />
                  <CardContent>
                    <Typography variant="body2" color="text.secondary">
                      Status: <span style={{ color: bot.status === 'running' ? 'green' : 'red' }}>{bot.status}</span>
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Balance: ${bot.account_info?.balance.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Equity: ${bot.account_info?.equity.toFixed(2)}
                    </Typography>
                    <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                      <Button size="small" variant="outlined" href={`/bots/${bot.id}`}>
                        View Details
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))
          ) : (
            <Grid item xs={12}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="body1" color="text.secondary">
                  No active bots. Start a bot to see it here.
                </Typography>
                <Button variant="contained" color="primary" sx={{ mt: 2 }} href="/bots/create">
                  Create Bot
                </Button>
              </Paper>
            </Grid>
          )}
        </Grid>
      </Box>
    </Box>
  );
};

export default Dashboard;
