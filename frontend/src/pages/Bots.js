import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Button,
  Grid,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import AddIcon from '@mui/icons-material/Add';
import { fetchBots, startBot, stopBot, deleteBot } from '../store/botsSlice';

const Bots = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { bots, loading } = useSelector((state) => state.bots);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [botToDelete, setBotToDelete] = useState(null);

  useEffect(() => {
    dispatch(fetchBots());
  }, [dispatch]);

  const handleViewBot = (id) => {
    navigate(`/bots/${id}`);
  };

  const handleStartBot = (id) => {
    dispatch(startBot(id));
  };

  const handleStopBot = (id) => {
    dispatch(stopBot(id));
  };

  const handleDeleteClick = (bot) => {
    setBotToDelete(bot);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (botToDelete) {
      dispatch(deleteBot(botToDelete.id));
      setDeleteDialogOpen(false);
      setBotToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setBotToDelete(null);
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

  if (loading && bots.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Trading Bots
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={() => navigate('/bots/create')}
        >
          Create Bot
        </Button>
      </Box>

      {bots.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>
            No trading bots found
          </Typography>
          <Typography variant="body1" color="textSecondary" paragraph>
            Create your first trading bot to get started with automated trading.
          </Typography>
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={() => navigate('/bots/create')}
          >
            Create Bot
          </Button>
        </Paper>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Strategy</TableCell>
                <TableCell>Symbol</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Balance</TableCell>
                <TableCell>P&L</TableCell>
                <TableCell>Last Update</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {bots.map((bot) => {
                const initialBalance = 10000; // Assuming default initial balance
                const pnl = (bot.account_info?.equity || 0) - initialBalance;
                const pnlPercentage = (pnl / initialBalance) * 100;

                return (
                  <TableRow key={bot.id}>
                    <TableCell>{bot.name}</TableCell>
                    <TableCell>{bot.strategy}</TableCell>
                    <TableCell>{bot.symbol}</TableCell>
                    <TableCell>
                      <Chip
                        label={bot.status}
                        color={getStatusColor(bot.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>${bot.account_info?.balance.toFixed(2)}</TableCell>
                    <TableCell sx={{ color: pnl >= 0 ? 'success.main' : 'error.main' }}>
                      ${pnl.toFixed(2)} ({pnlPercentage.toFixed(2)}%)
                    </TableCell>
                    <TableCell>
                      {new Date(bot.last_update).toLocaleString()}
                    </TableCell>
                    <TableCell align="center">
                      <IconButton
                        color="primary"
                        onClick={() => handleViewBot(bot.id)}
                        title="View Details"
                      >
                        <VisibilityIcon />
                      </IconButton>
                      {bot.status === 'running' ? (
                        <IconButton
                          color="warning"
                          onClick={() => handleStopBot(bot.id)}
                          title="Stop Bot"
                          disabled={loading}
                        >
                          <StopIcon />
                        </IconButton>
                      ) : (
                        <IconButton
                          color="success"
                          onClick={() => handleStartBot(bot.id)}
                          title="Start Bot"
                          disabled={loading || bot.status === 'error'}
                        >
                          <PlayArrowIcon />
                        </IconButton>
                      )}
                      <IconButton
                        color="error"
                        onClick={() => handleDeleteClick(bot)}
                        title="Delete Bot"
                        disabled={loading || bot.status === 'running'}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={handleDeleteCancel}
      >
        <DialogTitle>Delete Trading Bot</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the bot "{botToDelete?.name}"? This action cannot be undone.
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

export default Bots;
