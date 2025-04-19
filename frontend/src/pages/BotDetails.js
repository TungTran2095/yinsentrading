import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  CircularProgress,
  Chip,
  Divider,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Tab,
  Tabs
} from '@mui/material';
import { Line } from 'react-chartjs-2';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import { fetchBotById, startBot, stopBot, deleteBot } from '../store/botsSlice';

const BotDetails = () => {
  const { id } = useParams();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { currentBot, loading, error } = useSelector((state) => state.bots);
  const [tabValue, setTabValue] = useState(0);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

  useEffect(() => {
    dispatch(fetchBotById(id));
  }, [dispatch, id]);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleStartBot = () => {
    dispatch(startBot(id));
  };

  const handleStopBot = () => {
    dispatch(stopBot(id));
  };

  const handleDeleteClick = () => {
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    dispatch(deleteBot(id))
      .unwrap()
      .then(() => {
        navigate('/bots');
      });
    setDeleteDialogOpen(false);
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'stopped':
        return 'default';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  // Prepare chart data
  const preparePerformanceChartData = () => {
    if (!currentBot || !currentBot.performance_history) return null;

    return {
      labels: currentBot.performance_history.map(p => new Date(p.timestamp).toLocaleDateString()),
      datasets: [
        {
          label: 'Equity',
          data: currentBot.performance_history.map(p => p.equity),
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1,
          fill: false,
        },
      ],
    };
  };

  const prepareTradesChartData = () => {
    if (!currentBot || !currentBot.trades) return null;

    return {
      labels: currentBot.trades.map(t => new Date(t.timestamp).toLocaleDateString()),
      datasets: [
        {
          label: 'Profit/Loss',
          data: currentBot.trades.map(t => t.pnl),
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: currentBot.trades.map(t => 
            t.pnl >= 0 ? 'rgba(75, 192, 192, 0.5)' : 'rgba(255, 99, 132, 0.5)'
          ),
          type: 'bar',
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
        text: 'Bot Performance',
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      },
    },
  };

  if (loading && !currentBot) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom component="div">
          Error Loading Bot
        </Typography>
        <Paper sx={{ p: 3 }}>
          <Typography color="error">{error.message}</Typography>
          <Button variant="contained" onClick={() => navigate('/bots')} sx={{ mt: 2 }}>
            Back to Bots
          </Button>
        </Paper>
      </Box>
    );
  }

  if (!currentBot) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom component="div">
          Bot Not Found
        </Typography>
        <Paper sx={{ p: 3 }}>
          <Typography>The requested bot could not be found.</Typography>
          <Button variant="contained" onClick={() => navigate('/bots')} sx={{ mt: 2 }}>
            Back to Bots
          </Button>
        </Paper>
      </Box>
    );
  }

  const performanceChartData = preparePerformanceChartData();
  const tradesChartData = prepareTradesChartData();

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          {currentBot.name}
        </Typography>
        <Box>
          {currentBot.status === 'running' ? (
            <Button
              variant="contained"
              color="warning"
              startIcon={<StopIcon />}
              onClick={handleStopBot}
              disabled={loading}
              sx={{ mr: 1 }}
            >
              Stop Bot
            </Button>
          ) : (
            <Button
              variant="contained"
              color="success"
              startIcon={<PlayArrowIcon />}
              onClick={handleStartBot}
              disabled={loading || currentBot.status === 'error'}
              sx={{ mr: 1 }}
            >
              Start Bot
            </Button>
          )}
          <Button
            variant="outlined"
            color="primary"
            startIcon={<EditIcon />}
            onClick={() => navigate(`/bots/${id}/edit`)}
            sx={{ mr: 1 }}
          >
            Edit
          </Button>
          <Button
            variant="outlined"
            color="error"
            startIcon={<DeleteIcon />}
            onClick={handleDeleteClick}
            disabled={currentBot.status === 'running'}
          >
            Delete
          </Button>
        </Box>
      </Box>

      {/* Bot Summary */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle1" gutterBottom>
              Bot Information
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Status:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Chip
                  label={currentBot.status}
                  color={getStatusColor(currentBot.status)}
                  size="small"
                />
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Strategy:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">{currentBot.strategy}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Symbol:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">{currentBot.symbol}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Timeframe:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">{currentBot.timeframe}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Created:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  {new Date(currentBot.created_at).toLocaleString()}
                </Typography>
              </Grid>
            </Grid>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle1" gutterBottom>
              Account Information
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Initial Balance:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  ${currentBot.account_info?.initial_balance.toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Current Balance:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  ${currentBot.account_info?.balance.toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Equity:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  ${currentBot.account_info?.equity.toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Profit/Loss:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography
                  variant="body2"
                  color={
                    currentBot.account_info?.equity - currentBot.account_info?.initial_balance >= 0
                      ? 'success.main'
                      : 'error.main'
                  }
                >
                  $
                  {(
                    currentBot.account_info?.equity - currentBot.account_info?.initial_balance
                  ).toFixed(2)}{' '}
                  (
                  {(
                    ((currentBot.account_info?.equity - currentBot.account_info?.initial_balance) /
                      currentBot.account_info?.initial_balance) *
                    100
                  ).toFixed(2)}
                  %)
                </Typography>
              </Grid>
            </Grid>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle1" gutterBottom>
              Performance Metrics
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Total Trades:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">{currentBot.metrics?.total_trades || 0}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Win Rate:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  {currentBot.metrics?.win_rate ? `${(currentBot.metrics.win_rate * 100).toFixed(2)}%` : 'N/A'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Profit Factor:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  {currentBot.metrics?.profit_factor?.toFixed(2) || 'N/A'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Max Drawdown:
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="error.main">
                  {currentBot.metrics?.max_drawdown
                    ? `${(currentBot.metrics.max_drawdown * 100).toFixed(2)}%`
                    : 'N/A'}
                </Typography>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </Paper>

      {/* Tabs for different sections */}
      <Box sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="bot details tabs">
          <Tab label="Performance" />
          <Tab label="Trades" />
          <Tab label="Settings" />
          <Tab label="Logs" />
        </Tabs>
      </Box>

      {/* Tab Content */}
      <Box sx={{ mb: 3 }}>
        {/* Performance Tab */}
        {tabValue === 0 && (
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Performance Chart
            </Typography>
            {performanceChartData ? (
              <Box sx={{ height: 400 }}>
                <Line options={chartOptions} data={performanceChartData} />
              </Box>
            ) : (
              <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center', py: 5 }}>
                No performance data available yet.
              </Typography>
            )}
          </Paper>
        )}

        {/* Trades Tab */}
        {tabValue === 1 && (
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Trade History
            </Typography>
            {tradesChartData ? (
              <>
                <Box sx={{ height: 300, mb: 4 }}>
                  <Line options={chartOptions} data={tradesChartData} />
                </Box>
                <Typography variant="subtitle1" gutterBottom>
                  Recent Trades
                </Typography>
                {currentBot.trades && currentBot.trades.length > 0 ? (
                  <Box sx={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Date</th>
                          <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Type</th>
                          <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Price</th>
                          <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Size</th>
                          <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {currentBot.trades.slice(0, 10).map((trade, index) => (
                          <tr key={index}>
                            <td style={{ padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                              {new Date(trade.timestamp).toLocaleString()}
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                              <Chip
                                label={trade.type}
                                color={trade.type === 'buy' ? 'primary' : 'secondary'}
                                size="small"
                              />
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                              ${trade.price.toFixed(2)}
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                              {trade.size.toFixed(4)}
                            </td>
                            <td
                              style={{
                                padding: '8px',
                                borderBottom: '1px solid rgba(224, 224, 224, 1)',
                                color: trade.pnl >= 0 ? 'green' : 'red',
                              }}
                            >
                              ${trade.pnl.toFixed(2)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Box>
                ) : (
                  <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center', py: 3 }}>
                    No trades executed yet.
                  </Typography>
                )}
              </>
            ) : (
              <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center', py: 5 }}>
                No trade data available yet.
              </Typography>
            )}
          </Paper>
        )}

        {/* Settings Tab */}
        {tabValue === 2 && (
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Bot Settings
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardHeader title="Strategy Settings" />
                  <Divider />
                  <CardContent>
                    <Grid container spacing={1}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Strategy Type:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">{currentBot.config?.strategy?.type || 'N/A'}</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Symbol:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">{currentBot.config?.strategy?.symbol || 'N/A'}</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Timeframe:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">{currentBot.config?.strategy?.timeframe || 'N/A'}</Typography>
                      </Grid>
                      {currentBot.config?.strategy?.type === 'ensemble' && (
                        <>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Buy Threshold:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2">{currentBot.config?.strategy?.threshold_buy || 'N/A'}</Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Sell Threshold:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2">{currentBot.config?.strategy?.threshold_sell || 'N/A'}</Typography>
                          </Grid>
                        </>
                      )}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardHeader title="Risk Management" />
                  <Divider />
                  <CardContent>
                    <Grid container spacing={1}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Max Position Size:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          {currentBot.config?.risk?.max_position_size
                            ? `${(currentBot.config.risk.max_position_size * 100).toFixed(0)}%`
                            : 'N/A'}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Max Drawdown:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          {currentBot.config?.risk?.max_drawdown
                            ? `${(currentBot.config.risk.max_drawdown * 100).toFixed(0)}%`
                            : 'N/A'}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Stop Loss:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          {currentBot.config?.risk?.stop_loss_pct
                            ? `${(currentBot.config.risk.stop_loss_pct * 100).toFixed(0)}%`
                            : 'N/A'}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Take Profit:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          {currentBot.config?.risk?.take_profit_pct
                            ? `${(currentBot.config.risk.take_profit_pct * 100).toFixed(0)}%`
                            : 'N/A'}
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Paper>
        )}

        {/* Logs Tab */}
        {tabValue === 3 && (
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Bot Logs
            </Typography>
            <Box
              sx={{
                bgcolor: 'background.paper',
                p: 2,
                borderRadius: 1,
                maxHeight: 400,
                overflow: 'auto',
                fontFamily: 'monospace',
                fontSize: '0.875rem',
              }}
            >
              {currentBot.logs && currentBot.logs.length > 0 ? (
                currentBot.logs.map((log, index) => (
                  <Box key={index} sx={{ mb: 1 }}>
                    <Typography
                      variant="body2"
                      component="span"
                      sx={{ color: 'text.secondary', mr: 1 }}
                    >
                      {new Date(log.timestamp).toLocaleString()}
                    </Typography>
                    <Typography
                      variant="body2"
                      component="span"
                      sx={{
                        color:
                          log.level === 'ERROR'
                            ? 'error.main'
                            : log.level === 'WARNING'
                            ? 'warning.main'
                            : 'text.primary',
                      }}
                    >
                      [{log.level}] {log.message}
                    </Typography>
                  </Box>
                ))
              ) : (
                <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center', py: 3 }}>
                  No logs available.
                </Typography>
              )}
            </Box>
          </Paper>
        )}
      </Box>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={handleDeleteCancel}>
        <DialogTitle>Delete Trading Bot</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete this bot? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel} color="primary">
            Cancel
          </Button>
          <Button onClick={handleDeleteConfirm} color="error">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default BotDetails;
